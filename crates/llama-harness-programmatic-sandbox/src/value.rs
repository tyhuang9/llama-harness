use crate::{
    accounting::{
        array_framing_retained_bytes, checked_add, key_retained_bytes, measure_value,
        object_framing_retained_bytes, primitive_retained_bytes, serialized_string_len,
        string_retained_bytes, ValueMeasurement,
    },
    SandboxError, SandboxErrorCode, SandboxLimits,
};
use alloc::{string::String, sync::Arc, vec::Vec};
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct RuntimeValue {
    node: Arc<RuntimeNode>,
    measurement: ValueMeasurement,
}

pub(crate) enum RuntimeNode {
    Null,
    Bool(bool),
    Integer(i64),
    String(String),
    Array(Vec<RuntimeValue>),
    Object(Vec<(String, RuntimeValue)>),
}

enum BuildOperation<'a> {
    Visit(&'a Value),
    FinishArray(usize),
    FinishObject(Vec<&'a str>),
}

impl RuntimeValue {
    pub(crate) fn from_json(source: &Value, limits: &SandboxLimits) -> Result<Self, SandboxError> {
        let whole = measure_value(source, limits)?;
        let operation_capacity = whole
            .nodes
            .checked_mul(2)
            .ok_or_else(|| resource("value conversion work limit exceeded"))?;
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(operation_capacity)
            .map_err(|_| resource("value conversion allocation failed"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(whole.nodes)
            .map_err(|_| resource("value conversion allocation failed"))?;
        operations.push(BuildOperation::Visit(source));

        while let Some(operation) = operations.pop() {
            match operation {
                BuildOperation::Visit(value) => match value {
                    Value::Null => values.push(Self::null()),
                    Value::Bool(value) => values.push(Self::boolean(*value)),
                    Value::Number(value) => {
                        values.push(Self::integer(value.as_i64().ok_or_else(|| {
                            resource("JSON numbers must be signed 64-bit integers")
                        })?))
                    }
                    Value::String(value) => {
                        values.push(Self::string(clone_string(value)?)?);
                    }
                    Value::Array(items) => {
                        operations.push(BuildOperation::FinishArray(items.len()));
                        for item in items.iter().rev() {
                            operations.push(BuildOperation::Visit(item));
                        }
                    }
                    Value::Object(entries) => {
                        let mut keys = Vec::new();
                        keys.try_reserve_exact(entries.len())
                            .map_err(|_| resource("value conversion allocation failed"))?;
                        keys.extend(entries.keys().map(String::as_str));
                        operations.push(BuildOperation::FinishObject(keys));
                        for value in entries.values().rev() {
                            operations.push(BuildOperation::Visit(value));
                        }
                    }
                },
                BuildOperation::FinishArray(count) => {
                    let start = values
                        .len()
                        .checked_sub(count)
                        .ok_or_else(|| execution("value conversion stack underflow"))?;
                    let children = values.split_off(start);
                    values.push(Self::array(children)?);
                }
                BuildOperation::FinishObject(keys) => {
                    let start = values
                        .len()
                        .checked_sub(keys.len())
                        .ok_or_else(|| execution("value conversion stack underflow"))?;
                    let children = values.split_off(start);
                    let mut entries = Vec::new();
                    entries
                        .try_reserve_exact(keys.len())
                        .map_err(|_| resource("value conversion allocation failed"))?;
                    for (key, value) in keys.into_iter().zip(children) {
                        entries.push((clone_string(key)?, value));
                    }
                    values.push(Self::object(entries)?);
                }
            }
        }
        if values.len() != 1 {
            return Err(execution("value conversion did not produce one value"));
        }
        values
            .pop()
            .ok_or_else(|| execution("value conversion result is missing"))
    }

    pub(crate) fn null() -> Self {
        Self::scalar(RuntimeNode::Null, 4)
    }

    pub(crate) fn boolean(value: bool) -> Self {
        Self::scalar(RuntimeNode::Bool(value), if value { 4 } else { 5 })
    }

    pub(crate) fn integer(value: i64) -> Self {
        let serialized = if value == 0 {
            1
        } else {
            let magnitude = value.unsigned_abs();
            decimal_digits(magnitude) + usize::from(value < 0)
        };
        Self::scalar(RuntimeNode::Integer(value), serialized)
    }

    pub(crate) fn string(value: String) -> Result<Self, SandboxError> {
        let serialized = serialized_string_len(&value)?;
        Self::string_measured(value, serialized)
    }

    pub(crate) fn string_measured(value: String, serialized: usize) -> Result<Self, SandboxError> {
        let measurement = ValueMeasurement {
            retained: string_retained_bytes(value.capacity())?,
            serialized,
            nodes: 1,
            collection_items: 0,
            max_depth: 1,
        };
        Ok(Self {
            node: Arc::new(RuntimeNode::String(value)),
            measurement,
        })
    }

    pub(crate) fn array(values: Vec<Self>) -> Result<Self, SandboxError> {
        let mut measurement = ValueMeasurement {
            retained: array_framing_retained_bytes()?,
            serialized: checked_add(2, values.len().saturating_sub(1))?,
            nodes: 1,
            collection_items: values.len(),
            max_depth: 1,
        };
        for value in &values {
            accumulate_child(&mut measurement, value.measurement())?;
        }
        Ok(Self::array_measured(values, measurement))
    }

    pub(crate) fn array_measured(values: Vec<Self>, measurement: ValueMeasurement) -> Self {
        Self {
            node: Arc::new(RuntimeNode::Array(values)),
            measurement,
        }
    }

    pub(crate) fn object(values: Vec<(String, Self)>) -> Result<Self, SandboxError> {
        let mut measurement = ValueMeasurement {
            retained: object_framing_retained_bytes()?,
            serialized: checked_add(2, values.len().saturating_sub(1))?,
            nodes: 1,
            collection_items: values.len(),
            max_depth: 1,
        };
        for (key, value) in &values {
            measurement.retained =
                checked_add(measurement.retained, key_retained_bytes(key.capacity())?)?;
            measurement.serialized = checked_add(
                measurement.serialized,
                checked_add(serialized_string_len(key)?, 1)?,
            )?;
            accumulate_child(&mut measurement, value.measurement())?;
        }
        Ok(Self::object_measured(values, measurement))
    }

    pub(crate) fn object_measured(
        values: Vec<(String, Self)>,
        measurement: ValueMeasurement,
    ) -> Self {
        Self {
            node: Arc::new(RuntimeNode::Object(values)),
            measurement,
        }
    }

    pub(crate) fn response(ok: bool, output: Self) -> Result<Self, SandboxError> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(2)
            .map_err(|_| resource("response wrapper allocation failed"))?;
        entries.push((clone_string("ok")?, Self::boolean(ok)));
        entries.push((clone_string("output")?, output));
        Self::object(entries)
    }

    pub(crate) const fn measurement(&self) -> ValueMeasurement {
        self.measurement
    }

    pub(crate) fn node(&self) -> &RuntimeNode {
        &self.node
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self.node() {
            RuntimeNode::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_integer(&self) -> Option<i64> {
        match self.node() {
            RuntimeNode::Integer(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Self]> {
        match self.node() {
            RuntimeNode::Array(values) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn as_object(&self) -> Option<&[(String, Self)]> {
        match self.node() {
            RuntimeNode::Object(values) => Some(values),
            _ => None,
        }
    }

    fn scalar(node: RuntimeNode, serialized: usize) -> Self {
        Self {
            node: Arc::new(node),
            measurement: ValueMeasurement {
                retained: primitive_retained_bytes(),
                serialized,
                nodes: 1,
                collection_items: 0,
                max_depth: 1,
            },
        }
    }
}

fn accumulate_child(
    measurement: &mut ValueMeasurement,
    child: ValueMeasurement,
) -> Result<(), SandboxError> {
    measurement.retained = checked_add(measurement.retained, child.retained)?;
    measurement.serialized = checked_add(measurement.serialized, child.serialized)?;
    measurement.nodes = checked_add(measurement.nodes, child.nodes)?;
    measurement.collection_items =
        checked_add(measurement.collection_items, child.collection_items)?;
    measurement.max_depth = measurement.max_depth.max(
        child
            .max_depth
            .checked_add(1)
            .ok_or_else(|| resource("value nesting limit exceeded"))?,
    );
    Ok(())
}

fn decimal_digits(mut value: u64) -> usize {
    let mut digits = 1usize;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

fn clone_string(source: &str) -> Result<String, SandboxError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(source.len())
        .map_err(|_| resource("string clone allocation failed"))?;
    cloned.push_str(source);
    Ok(cloned)
}

fn resource(message: &'static str) -> SandboxError {
    SandboxError::new(SandboxErrorCode::ResourceLimit, message)
}

fn execution(message: &'static str) -> SandboxError {
    SandboxError::new(SandboxErrorCode::Execution, message)
}
