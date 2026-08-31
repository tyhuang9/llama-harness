use serde_json::Value;

/// Replacement text used by the default redaction configuration.
pub const REDACTED_VALUE: &str = "[REDACTED]";

/// Redaction rules applied before any event or raw payload is serialized for persistence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedactionConfig {
    /// Case-insensitive exact key names or delimited token sequences whose
    /// values must be redacted recursively.
    pub key_fragments: Vec<String>,
    /// Literal secret values to remove anywhere they occur in a string.
    pub secret_values: Vec<String>,
    /// Text substituted for redacted keys and secret values.
    pub replacement: String,
}

impl Default for RedactionConfig {
    fn default() -> Self {
        Self {
            key_fragments: vec![
                "authorization".into(),
                "cookie".into(),
                "password".into(),
                "secret".into(),
                "token".into(),
                "api_key".into(),
                "apikey".into(),
                "program".into(),
                "ast".into(),
                "bytecode".into(),
                "constant".into(),
                "locals".into(),
                "source".into(),
            ],
            secret_values: vec![],
            replacement: REDACTED_VALUE.into(),
        }
    }
}

impl RedactionConfig {
    /// Recursively redacts matching object keys and configured secret values.
    pub fn redact(&self, value: &Value) -> Value {
        match value {
            Value::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(key, value)| {
                        let redacted = if self.redacts_key(key) {
                            Value::String(self.replacement.clone())
                        } else {
                            self.redact(value)
                        };
                        (key.clone(), redacted)
                    })
                    .collect(),
            ),
            Value::Array(values) => {
                Value::Array(values.iter().map(|value| self.redact(value)).collect())
            }
            Value::String(value) => Value::String(self.redact_string(value)),
            value => value.clone(),
        }
    }

    fn redacts_key(&self, key: &str) -> bool {
        let normalized = key.to_ascii_lowercase();
        let tokens = key_tokens(key);
        self.key_fragments.iter().any(|configured| {
            let configured = configured.trim().to_ascii_lowercase();
            if configured.is_empty() {
                return false;
            }
            if normalized == configured {
                return true;
            }
            let configured_tokens = key_tokens(&configured);
            !configured_tokens.is_empty()
                && tokens
                    .windows(configured_tokens.len())
                    .any(|candidate| candidate == configured_tokens.as_slice())
        })
    }

    fn redact_string(&self, value: &str) -> String {
        self.secret_values
            .iter()
            .filter(|secret| !secret.is_empty())
            .fold(value.to_owned(), |value, secret| {
                value.replace(secret, &self.replacement)
            })
    }
}

fn key_tokens(key: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for component in key
        .split(|character: char| character == '_' || !character.is_ascii_alphanumeric())
        .filter(|component| !component.is_empty())
    {
        let bytes = component.as_bytes();
        let mut start = 0;
        for index in 1..bytes.len() {
            let previous = bytes[index - 1];
            let current = bytes[index];
            let next = bytes.get(index + 1).copied();
            let camel_case = previous.is_ascii_lowercase() && current.is_ascii_uppercase();
            let acronym = previous.is_ascii_uppercase()
                && current.is_ascii_uppercase()
                && next.is_some_and(|next| next.is_ascii_lowercase());
            if camel_case || acronym {
                tokens.push(component[start..index].to_ascii_lowercase());
                start = index;
            }
        }
        tokens.push(component[start..].to_ascii_lowercase());
    }
    tokens
}
