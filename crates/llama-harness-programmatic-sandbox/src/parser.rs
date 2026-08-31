use crate::{
    accounting::{key_retained_bytes, primitive_retained_bytes, string_retained_bytes},
    ast::deserialize_program,
    Expression, ObjectEntry, Program, SandboxError, SandboxErrorCode, SandboxLimits, Statement,
    MAX_ATOMIC_KEY_BYTES, MAX_ATOMIC_STRING_BYTES, PROGRAM_VERSION_V1,
};
use alloc::{collections::BTreeSet, string::ToString, vec::Vec};

pub(crate) fn parse_program(input: &[u8], limits: &SandboxLimits) -> Result<Program, SandboxError> {
    limits.validate()?;
    if input.len() > limits.max_program_bytes {
        return Err(resource("program byte limit exceeded"));
    }
    core::str::from_utf8(input).map_err(|_| invalid("program must be valid UTF-8"))?;
    validate_json_nesting(input, limits.max_nesting)?;
    let program = deserialize_program(input)
        .map_err(|_| invalid("program does not match the strict V1 JSON schema"))?;
    validate_ast(&program, limits)?;
    Ok(program)
}

fn validate_json_nesting(input: &[u8], max_depth: usize) -> Result<(), SandboxError> {
    let mut delimiters = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for byte in input.iter().copied() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                delimiters
                    .try_reserve(1)
                    .map_err(|_| resource("JSON nesting validation allocation failed"))?;
                delimiters.push(byte);
                if delimiters.len() > max_depth {
                    return Err(resource("JSON nesting limit exceeded"));
                }
            }
            b'}' if delimiters.pop() != Some(b'{') => {
                return Err(invalid("program JSON delimiters are unbalanced"));
            }
            b']' if delimiters.pop() != Some(b'[') => {
                return Err(invalid("program JSON delimiters are unbalanced"));
            }
            b'}' | b']' => {}
            _ => {}
        }
    }
    if in_string || !delimiters.is_empty() {
        return Err(invalid("program JSON is incomplete"));
    }
    Ok(())
}

enum Node<'a> {
    Statement(&'a Statement, usize),
    Expression(&'a Expression, usize),
}

pub(crate) fn validate_ast(program: &Program, limits: &SandboxLimits) -> Result<(), SandboxError> {
    limits.validate()?;
    if program.version != PROGRAM_VERSION_V1 {
        return Err(invalid("unsupported program version"));
    }
    let mut stack = Vec::new();
    stack
        .try_reserve(program.body.len())
        .map_err(|_| resource("program validation allocation failed"))?;
    for statement in program.body.iter().rev() {
        stack.push(Node::Statement(statement, 1));
    }
    let mut nodes = 1usize;
    let mut constant_bytes = 0usize;
    let mut declared_locals = BTreeSet::new();

    while let Some(node) = stack.pop() {
        nodes = nodes
            .checked_add(1)
            .ok_or_else(|| resource("AST node limit exceeded"))?;
        if nodes > limits.max_ast_nodes {
            return Err(resource("AST node limit exceeded"));
        }
        match node {
            Node::Statement(statement, depth) => {
                ensure_depth(depth, limits)?;
                match statement {
                    Statement::Let { name, value } => {
                        declare(name, &mut declared_locals, limits)?;
                        stack.push(Node::Expression(value, depth + 1));
                    }
                    Statement::Branch {
                        condition,
                        then_body,
                        else_body,
                    } => {
                        stack.push(Node::Expression(condition, depth + 1));
                        push_statements(&mut stack, then_body, depth + 1)?;
                        push_statements(&mut stack, else_body, depth + 1)?;
                    }
                    Statement::ForEach {
                        item,
                        collection,
                        max_iterations,
                        body,
                    } => {
                        validate_name(item)?;
                        validate_bound(
                            *max_iterations as usize,
                            limits.max_loop_iterations,
                            "loop",
                        )?;
                        stack.push(Node::Expression(collection, depth + 1));
                        push_statements(&mut stack, body, depth + 1)?;
                    }
                    Statement::Map {
                        name,
                        item,
                        collection,
                        max_items,
                        value,
                    }
                    | Statement::Filter {
                        name,
                        item,
                        collection,
                        max_items,
                        predicate: value,
                    } => {
                        declare(name, &mut declared_locals, limits)?;
                        validate_name(item)?;
                        validate_bound(
                            *max_items as usize,
                            limits.max_collection_items,
                            "collection",
                        )?;
                        stack.push(Node::Expression(collection, depth + 1));
                        stack.push(Node::Expression(value, depth + 1));
                    }
                    Statement::Reduce {
                        name,
                        item,
                        accumulator,
                        collection,
                        max_items,
                        initial,
                        value,
                    } => {
                        declare(name, &mut declared_locals, limits)?;
                        validate_name(item)?;
                        validate_name(accumulator)?;
                        if item == accumulator {
                            return Err(invalid("reduce item and accumulator names must differ"));
                        }
                        validate_bound(
                            *max_items as usize,
                            limits.max_collection_items,
                            "collection",
                        )?;
                        stack.push(Node::Expression(collection, depth + 1));
                        stack.push(Node::Expression(initial, depth + 1));
                        stack.push(Node::Expression(value, depth + 1));
                    }
                    Statement::Invoke {
                        name,
                        tool_id,
                        arguments,
                    } => {
                        declare(name, &mut declared_locals, limits)?;
                        validate_tool_id(tool_id)?;
                        stack.push(Node::Expression(arguments, depth + 1));
                    }
                    Statement::FanOut {
                        name,
                        tool_id,
                        item,
                        collection,
                        max_calls,
                        arguments,
                    } => {
                        declare(name, &mut declared_locals, limits)?;
                        validate_tool_id(tool_id)?;
                        validate_name(item)?;
                        validate_bound(*max_calls as usize, limits.max_fanout, "fan-out")?;
                        stack.push(Node::Expression(collection, depth + 1));
                        stack.push(Node::Expression(arguments, depth + 1));
                    }
                    Statement::Return { value } => stack.push(Node::Expression(value, depth + 1)),
                }
            }
            Node::Expression(expression, depth) => {
                ensure_depth(depth, limits)?;
                match expression {
                    Expression::Null | Expression::Boolean { .. } | Expression::Integer { .. } => {
                        add_constant(&mut constant_bytes, primitive_retained_bytes(), limits)?;
                    }
                    Expression::String { value } => add_constant(
                        &mut constant_bytes,
                        string_retained_bytes(value.capacity())?,
                        limits,
                    )?,
                    Expression::Variable { name } => validate_name(name)?,
                    Expression::Path { value, pointer } => {
                        validate_pointer(pointer)?;
                        add_constant(
                            &mut constant_bytes,
                            string_retained_bytes(pointer.capacity())?,
                            limits,
                        )?;
                        stack.push(Node::Expression(value, depth + 1));
                    }
                    Expression::Array { items } => {
                        validate_count(items.len(), limits.max_collection_items, "collection")?;
                        push_expressions(&mut stack, items, depth + 1)?;
                    }
                    Expression::Object { entries } => {
                        validate_count(entries.len(), limits.max_collection_items, "collection")?;
                        let mut keys = BTreeSet::new();
                        for ObjectEntry { key, value } in entries.iter().rev() {
                            if key.len() > MAX_ATOMIC_KEY_BYTES {
                                return Err(invalid("object keys must contain at most 64 bytes"));
                            }
                            if !keys.insert(key.as_str()) {
                                return Err(invalid("object keys must be unique"));
                            }
                            add_constant(
                                &mut constant_bytes,
                                key_retained_bytes(key.capacity())?,
                                limits,
                            )?;
                            stack.push(Node::Expression(value, depth + 1));
                        }
                    }
                    Expression::Binary { left, right, .. } => {
                        stack.push(Node::Expression(right, depth + 1));
                        stack.push(Node::Expression(left, depth + 1));
                    }
                    Expression::Unary { value, .. } => {
                        stack.push(Node::Expression(value, depth + 1))
                    }
                }
            }
        }
    }
    Ok(())
}

fn push_statements<'a>(
    stack: &mut Vec<Node<'a>>,
    body: &'a [Statement],
    depth: usize,
) -> Result<(), SandboxError> {
    stack
        .try_reserve(body.len())
        .map_err(|_| resource("program validation allocation failed"))?;
    for statement in body.iter().rev() {
        stack.push(Node::Statement(statement, depth));
    }
    Ok(())
}

fn push_expressions<'a>(
    stack: &mut Vec<Node<'a>>,
    expressions: &'a [Expression],
    depth: usize,
) -> Result<(), SandboxError> {
    stack
        .try_reserve(expressions.len())
        .map_err(|_| resource("program validation allocation failed"))?;
    for expression in expressions.iter().rev() {
        stack.push(Node::Expression(expression, depth));
    }
    Ok(())
}

fn declare<'a>(
    name: &'a str,
    locals: &mut BTreeSet<&'a str>,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    validate_name(name)?;
    if !locals.insert(name) {
        return Err(invalid("immutable local names must be unique"));
    }
    if locals.len() > limits.max_locals {
        return Err(resource("local binding limit exceeded"));
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), SandboxError> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
    {
        return Err(invalid(
            "local names must contain 1..=128 ASCII identifier bytes",
        ));
    }
    Ok(())
}

fn validate_tool_id(tool_id: &str) -> Result<(), SandboxError> {
    if tool_id.is_empty() || tool_id.len() > MAX_ATOMIC_STRING_BYTES {
        return Err(invalid("tool IDs must contain 1..=64 bytes"));
    }
    Ok(())
}

fn validate_pointer(pointer: &str) -> Result<(), SandboxError> {
    if pointer.len() > 1024 || (!pointer.is_empty() && !pointer.starts_with('/')) {
        return Err(invalid(
            "JSON pointers must be empty or slash-prefixed and at most 1024 bytes",
        ));
    }
    let bytes = pointer.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            index += 1;
            if index == bytes.len() || !matches!(bytes[index], b'0' | b'1') {
                return Err(invalid("JSON pointers must use valid RFC 6901 escapes"));
            }
        }
        index += 1;
    }
    Ok(())
}

fn validate_bound(value: usize, max: usize, label: &str) -> Result<(), SandboxError> {
    if value == 0 || value > max {
        return Err(resource(alloc::format!(
            "{label} bound must be nonzero and within the effective limit"
        )));
    }
    Ok(())
}

fn validate_count(value: usize, max: usize, label: &str) -> Result<(), SandboxError> {
    if value > max {
        return Err(resource(alloc::format!(
            "{label} size exceeds the effective limit"
        )));
    }
    Ok(())
}

fn ensure_depth(depth: usize, limits: &SandboxLimits) -> Result<(), SandboxError> {
    if depth > limits.max_nesting {
        return Err(resource("language nesting limit exceeded"));
    }
    Ok(())
}

fn add_constant(
    total: &mut usize,
    bytes: usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| resource("constant byte limit exceeded"))?;
    if *total > limits.max_constant_bytes {
        return Err(resource("constant byte limit exceeded"));
    }
    Ok(())
}

fn invalid(message: impl ToString) -> SandboxError {
    SandboxError::new(SandboxErrorCode::InvalidProgram, message.to_string())
}

fn resource(message: impl ToString) -> SandboxError {
    SandboxError::new(SandboxErrorCode::ResourceLimit, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HARD_LIMITS;
    use serde_json::json;

    fn parse(value: serde_json::Value) -> Result<Program, SandboxError> {
        Program::from_json(
            &serde_json::to_vec(&value).unwrap(),
            &SandboxLimits::default(),
        )
    }

    #[test]
    fn accepts_strict_v1_program() {
        let program = parse(json!({
            "version": 1,
            "body": [{"kind":"return","value":{"kind":"integer","value":7}}]
        }))
        .unwrap();
        assert_eq!(program.version, 1);
        assert_eq!(program.statement_count(), 1);
    }

    #[test]
    fn rejects_unknown_duplicate_and_float_fields() {
        let limits = SandboxLimits::default();
        for raw in [
            br#"{"version":1,"extra":true,"body":[]}"#.as_slice(),
            br#"{"version":1,"version":1,"body":[]}"#.as_slice(),
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"integer","value":1.5}}]}"#
                .as_slice(),
        ] {
            assert_eq!(
                Program::from_json(raw, &limits).unwrap_err().code(),
                SandboxErrorCode::InvalidProgram
            );
        }
    }

    #[test]
    fn rejects_nested_duplicates_trailing_data_and_unbalanced_json() {
        let limits = SandboxLimits::default();
        for raw in [
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"integer","value":1,"value":2}}]}"#.as_slice(),
            br#"{"version":1,"body":[]} true"#.as_slice(),
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"null"}}]"#.as_slice(),
            br#"{"version":1,"body":[]]}"#.as_slice(),
            br#"{"version":1,"body":[}"#.as_slice(),
        ] {
            assert_eq!(
                Program::from_json(raw, &limits).unwrap_err().code(),
                SandboxErrorCode::InvalidProgram
            );
        }
    }

    #[test]
    fn rejects_invalid_version_utf8_depth_and_size() {
        assert!(parse(json!({"version":2,"body":[]})).is_err());
        assert!(Program::from_json(&[0xff], &SandboxLimits::default()).is_err());
        let mut limits = SandboxLimits {
            max_nesting: 2,
            ..SandboxLimits::default()
        };
        assert!(Program::from_json(br#"{"version":1,"body":[[]]}"#, &limits).is_err());
        limits = SandboxLimits::default();
        limits.max_program_bytes = 8;
        assert!(Program::from_json(br#"{"version":1,"body":[]}"#, &limits).is_err());
    }

    #[test]
    fn validates_bounds_names_objects_and_pointers() {
        for value in [
            json!({"version":1,"body":[{"kind":"let","name":"","value":{"kind":"null"}}]}),
            json!({"version":1,"body":[
                {"kind":"let","name":"x","value":{"kind":"null"}},
                {"kind":"let","name":"x","value":{"kind":"null"}}
            ]}),
            json!({"version":1,"body":[{"kind":"return","value":{"kind":"path","value":{"kind":"null"},"pointer":"/~2"}}]}),
            json!({"version":1,"body":[{"kind":"return","value":{"kind":"object","entries":[
                {"key":"x","value":{"kind":"null"}}, {"key":"x","value":{"kind":"null"}}
            ]}}]}),
            json!({"version":1,"body":[{"kind":"fan_out","name":"x","tool_id":"read","item":"i","collection":{"kind":"array","items":[]},"max_calls":9,"arguments":{"kind":"null"}}]}),
        ] {
            assert!(parse(value).is_err());
        }
    }

    #[test]
    fn limit_validation_is_fail_closed() {
        let mut limits = SandboxLimits {
            max_fuel: 0,
            ..SandboxLimits::default()
        };
        assert_eq!(
            limits.validate().unwrap_err().code(),
            SandboxErrorCode::InvalidLimits
        );
        limits = HARD_LIMITS;
        limits.max_ast_nodes += 1;
        assert_eq!(
            limits.validate().unwrap_err().code(),
            SandboxErrorCode::InvalidLimits
        );
    }

    #[test]
    fn parser_enforces_exact_program_constant_local_and_collection_boundaries() {
        let raw = br#"{"version":1,"body":[]}"#;
        let exact_program_limit = SandboxLimits {
            max_program_bytes: raw.len(),
            ..SandboxLimits::default()
        };
        assert!(Program::from_json(raw, &exact_program_limit).is_ok());
        let below_program_limit = SandboxLimits {
            max_program_bytes: raw.len() - 1,
            ..SandboxLimits::default()
        };
        assert_eq!(
            Program::from_json(raw, &below_program_limit)
                .unwrap_err()
                .code(),
            SandboxErrorCode::ResourceLimit
        );

        let constant = json!({"version":1,"body":[{
            "kind":"return","value":{"kind":"string","value":"abc"}
        }]});
        let exact_constant_bytes = string_retained_bytes(3).unwrap();
        for (limit, accepted) in [
            (exact_constant_bytes - 1, false),
            (exact_constant_bytes, true),
            (exact_constant_bytes + 1, true),
        ] {
            let limits = SandboxLimits {
                max_constant_bytes: limit,
                ..SandboxLimits::default()
            };
            let parsed = Program::from_json(&serde_json::to_vec(&constant).unwrap(), &limits);
            if accepted {
                assert!(parsed.is_ok());
            } else {
                assert_eq!(parsed.unwrap_err().code(), SandboxErrorCode::ResourceLimit);
            }
        }

        let locals = json!({"version":1,"body":[
            {"kind":"let","name":"first","value":{"kind":"null"}},
            {"kind":"let","name":"second","value":{"kind":"null"}},
            {"kind":"return","value":{"kind":"null"}}
        ]});
        let locals_bytes = serde_json::to_vec(&locals).unwrap();
        assert_eq!(
            Program::from_json(
                &locals_bytes,
                &SandboxLimits {
                    max_locals: 1,
                    ..SandboxLimits::default()
                }
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );
        assert!(Program::from_json(
            &locals_bytes,
            &SandboxLimits {
                max_locals: 2,
                ..SandboxLimits::default()
            }
        )
        .is_ok());

        let collection = json!({"version":1,"body":[{
            "kind":"return","value":{"kind":"array","items":[
                {"kind":"integer","value":1},{"kind":"integer","value":2}
            ]}
        }]});
        let collection_bytes = serde_json::to_vec(&collection).unwrap();
        assert_eq!(
            Program::from_json(
                &collection_bytes,
                &SandboxLimits {
                    max_collection_items: 1,
                    ..SandboxLimits::default()
                }
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );
        assert!(Program::from_json(
            &collection_bytes,
            &SandboxLimits {
                max_collection_items: 2,
                ..SandboxLimits::default()
            }
        )
        .is_ok());

        let loop_program = json!({"version":1,"body":[
            {"kind":"for_each","item":"item","collection":{"kind":"array","items":[]},"max_iterations":2,"body":[]},
            {"kind":"return","value":{"kind":"null"}}
        ]});
        let loop_bytes = serde_json::to_vec(&loop_program).unwrap();
        assert_eq!(
            Program::from_json(
                &loop_bytes,
                &SandboxLimits {
                    max_loop_iterations: 1,
                    ..SandboxLimits::default()
                }
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );
        assert!(Program::from_json(
            &loop_bytes,
            &SandboxLimits {
                max_loop_iterations: 2,
                ..SandboxLimits::default()
            }
        )
        .is_ok());
    }
}
