use crate::{SandboxError, SandboxErrorCode, SandboxLimits, MAX_ATOMIC_KEY_BYTES};
use alloc::{string::ToString, vec::Vec};
use core::mem::size_of;
use serde_json::Value;

// These charges intentionally describe a conservative logical allocation
// model instead of allocator-specific usable sizes. Every crate boundary uses
// this same model, so a value cannot become cheaper by moving from source,
// through bytecode, into the VM, or across a tool-response boundary.
const ALLOCATION_OVERHEAD: usize = size_of::<usize>() * 3;
const MAP_ENTRY_OVERHEAD: usize = size_of::<usize>() * 5;
const STRING_HEADER: usize = size_of::<alloc::string::String>();
const VALUE_SLOT: usize = size_of::<Value>();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ValueMeasurement {
    pub(crate) retained: usize,
    pub(crate) serialized: usize,
    pub(crate) nodes: usize,
    pub(crate) collection_items: usize,
    pub(crate) max_depth: usize,
}

pub(crate) const fn primitive_retained_bytes() -> usize {
    VALUE_SLOT
}

pub(crate) fn string_retained_bytes(bytes: usize) -> Result<usize, SandboxError> {
    checked_add(VALUE_SLOT, checked_add(ALLOCATION_OVERHEAD, bytes)?)
}

pub(crate) fn key_retained_bytes(bytes: usize) -> Result<usize, SandboxError> {
    checked_add(
        MAP_ENTRY_OVERHEAD,
        checked_add(STRING_HEADER, checked_add(ALLOCATION_OVERHEAD, bytes)?)?,
    )
}

pub(crate) fn array_framing_retained_bytes() -> Result<usize, SandboxError> {
    checked_add(VALUE_SLOT, ALLOCATION_OVERHEAD)
}

pub(crate) fn object_framing_retained_bytes() -> Result<usize, SandboxError> {
    checked_add(VALUE_SLOT, ALLOCATION_OVERHEAD)
}

pub(crate) fn vector_allocation_bytes<T>(capacity: usize) -> Result<usize, SandboxError> {
    if capacity == 0 {
        return Ok(0);
    }
    checked_add(
        ALLOCATION_OVERHEAD,
        size_of::<T>()
            .checked_mul(capacity)
            .ok_or_else(|| resource("value byte limit exceeded"))?,
    )
}

/// Iteratively validates and measures one untrusted JSON value.
///
/// This is deliberately the only whole-value size/depth/shape walker used by
/// the compiler verifier, VM accounting, response boundary, and output limit.
/// It never calls serde serialization or recursively visits attacker-owned
/// values.
pub(crate) fn measure_value(
    value: &Value,
    limits: &SandboxLimits,
) -> Result<ValueMeasurement, SandboxError> {
    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|_| resource("value measurement allocation failed"))?;
    stack.push((value, 1usize));

    let mut measurement = ValueMeasurement::default();
    while let Some((current, depth)) = stack.pop() {
        if depth > limits.max_nesting {
            return Err(resource("value nesting limit exceeded"));
        }
        measurement.max_depth = measurement.max_depth.max(depth);
        measurement.nodes = checked_add(measurement.nodes, 1)?;
        measurement.retained = checked_add(measurement.retained, VALUE_SLOT)?;

        match current {
            Value::Null => measurement.serialized = checked_add(measurement.serialized, 4)?,
            Value::Bool(true) => measurement.serialized = checked_add(measurement.serialized, 4)?,
            Value::Bool(false) => measurement.serialized = checked_add(measurement.serialized, 5)?,
            Value::Number(number) => {
                if number.as_i64().is_none() {
                    return Err(resource("JSON numbers must be signed 64-bit integers"));
                }
                measurement.serialized = checked_add(
                    measurement.serialized,
                    number.as_i64().unwrap_or_default().to_string().len(),
                )?;
            }
            Value::String(string) => {
                measurement.retained = checked_add(
                    measurement.retained,
                    checked_add(ALLOCATION_OVERHEAD, string.capacity())?,
                )?;
                measurement.serialized =
                    checked_add(measurement.serialized, serialized_string_len(string)?)?;
            }
            Value::Array(values) => {
                validate_collection(values.len(), limits)?;
                measurement.collection_items =
                    checked_add(measurement.collection_items, values.len())?;
                measurement.retained = checked_add(
                    measurement.retained,
                    checked_add(
                        ALLOCATION_OVERHEAD,
                        VALUE_SLOT
                            .checked_mul(values.capacity().saturating_sub(values.len()))
                            .ok_or_else(|| resource("value retained byte limit exceeded"))?,
                    )?,
                )?;
                measurement.serialized = checked_add(
                    measurement.serialized,
                    checked_add(2, values.len().saturating_sub(1))?,
                )?;
                stack
                    .try_reserve(values.len())
                    .map_err(|_| resource("value measurement allocation failed"))?;
                for child in values.iter().rev() {
                    stack.push((child, depth.saturating_add(1)));
                }
            }
            Value::Object(values) => {
                validate_collection(values.len(), limits)?;
                measurement.collection_items =
                    checked_add(measurement.collection_items, values.len())?;
                measurement.retained = checked_add(measurement.retained, ALLOCATION_OVERHEAD)?;
                measurement.serialized = checked_add(
                    measurement.serialized,
                    checked_add(2, values.len().saturating_sub(1))?,
                )?;
                stack
                    .try_reserve(values.len())
                    .map_err(|_| resource("value measurement allocation failed"))?;
                for (key, child) in values.iter().rev() {
                    if key.len() > MAX_ATOMIC_KEY_BYTES {
                        return Err(resource("object keys must contain at most 64 bytes"));
                    }
                    measurement.retained =
                        checked_add(measurement.retained, key_retained_bytes(key.capacity())?)?;
                    measurement.serialized = checked_add(
                        measurement.serialized,
                        checked_add(serialized_string_len(key)?, 1)?,
                    )?;
                    stack.push((child, depth.saturating_add(1)));
                }
            }
        }
        if measurement.retained > limits.max_live_bytes
            || measurement.retained > limits.max_cumulative_bytes
        {
            return Err(resource("value retained byte limit exceeded"));
        }
    }
    Ok(measurement)
}

pub(crate) fn serialized_string_len(value: &str) -> Result<usize, SandboxError> {
    let mut total = 2usize;
    for byte in value.bytes() {
        total = checked_add(
            total,
            match byte {
                b'"' | b'\\' => 2,
                0x00..=0x1f => 6,
                _ => 1,
            },
        )?;
    }
    Ok(total)
}

pub(crate) fn checked_add(total: usize, additional: usize) -> Result<usize, SandboxError> {
    total
        .checked_add(additional)
        .ok_or_else(|| resource("value byte limit exceeded"))
}

fn validate_collection(count: usize, limits: &SandboxLimits) -> Result<(), SandboxError> {
    if count > limits.max_collection_items {
        Err(resource("collection item limit exceeded"))
    } else {
        Ok(())
    }
}

fn resource(message: &'static str) -> SandboxError {
    SandboxError::new(SandboxErrorCode::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use serde_json::json;

    #[test]
    fn retained_and_serialized_boundaries_are_inclusive() {
        let value = json!({"key":[null,true,7,"value"]});
        let baseline = measure_value(&value, &SandboxLimits::default()).unwrap();
        assert_eq!(
            baseline.serialized,
            serde_json::to_vec(&value).unwrap().len()
        );
        for (limit, accepted) in [
            (baseline.retained - 1, false),
            (baseline.retained, true),
            (baseline.retained + 1, true),
        ] {
            let limits = SandboxLimits {
                max_live_bytes: limit,
                max_cumulative_bytes: limit,
                ..SandboxLimits::default()
            };
            assert_eq!(measure_value(&value, &limits).is_ok(), accepted);
        }
    }

    #[test]
    fn empty_object_keys_are_measured_with_their_object_member_framing() {
        let value = json!({"":null});
        let measurement = measure_value(&value, &SandboxLimits::default()).unwrap();
        assert_eq!(
            measurement.serialized,
            serde_json::to_vec(&value).unwrap().len()
        );
        assert!(
            measurement.retained >= primitive_retained_bytes() + key_retained_bytes(0).unwrap()
        );
    }

    #[test]
    fn depth_collection_and_numeric_domains_are_checked_iteratively() {
        let mut at_depth = Value::Null;
        for _ in 1..4 {
            at_depth = Value::Array(vec![at_depth]);
        }
        let limits = SandboxLimits {
            max_nesting: 4,
            max_collection_items: 2,
            ..SandboxLimits::default()
        };
        assert_eq!(measure_value(&at_depth, &limits).unwrap().max_depth, 4);
        let too_deep = Value::Array(vec![at_depth]);
        assert_eq!(
            measure_value(&too_deep, &limits).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        assert_eq!(
            measure_value(&Value::Array(vec![Value::Null; 3]), &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::ResourceLimit
        );
        assert_eq!(
            measure_value(&json!(1.5), &limits).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );

        // A moderately deep value exercises the explicit work stack. Dropping
        // it at this bounded depth is also safe on test platforms.
        let mut hostile = Value::Null;
        for _ in 0..limits.max_nesting + 1 {
            hostile = Value::Array(Vec::from([hostile]));
        }
        assert!(measure_value(&hostile, &limits).is_err());
    }

    #[test]
    fn retained_measurement_counts_host_controlled_spare_capacity() {
        let mut string = alloc::string::String::new();
        string.try_reserve_exact(4_096).unwrap();
        string.push('x');
        let string_value = Value::String(string);
        let string_measurement = measure_value(&string_value, &SandboxLimits::default()).unwrap();
        assert!(string_measurement.retained >= primitive_retained_bytes() + 4_096);

        let mut items = Vec::new();
        items.try_reserve_exact(64).unwrap();
        items.push(Value::Null);
        let array_value = Value::Array(items);
        let array_measurement = measure_value(&array_value, &SandboxLimits::default()).unwrap();
        assert!(array_measurement.retained >= primitive_retained_bytes() + size_of::<Value>() * 64);
    }
}
