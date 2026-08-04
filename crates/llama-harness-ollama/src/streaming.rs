use crate::wire::{tool_calls, usage, ChatResponse};
use async_stream::stream;
use futures_util::{Stream, StreamExt};
use llama_harness_core::{HarnessError, ToolCall, Usage};
use reqwest::Response;
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

pub type OllamaEventStream =
    Pin<Box<dyn Stream<Item = Result<OllamaStreamEvent, HarnessError>> + Send>>;

#[derive(Clone, Debug, PartialEq)]
pub enum OllamaStreamEvent {
    TextDelta { content: String },
    ToolCall { call: ToolCall },
    Completed { model: String, usage: Usage },
}

pub(crate) fn stream_response(
    response: Response,
    cancellation: CancellationToken,
    max_stream_bytes: usize,
    max_line_bytes: usize,
) -> OllamaEventStream {
    Box::pin(stream! {
        let mut chunks = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut received_bytes = 0_usize;
        loop {
            let next = tokio::select! {
                _ = cancellation.cancelled() => {
                    yield Err(HarnessError::Cancelled);
                    return;
                }
                chunk = chunks.next() => chunk,
            };
            let Some(chunk) = next else {
                break;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(HarnessError::RetryableProvider(format!("Ollama stream failed: {error}")));
                    return;
                }
            };
            received_bytes = received_bytes.saturating_add(chunk.len());
            if received_bytes > max_stream_bytes {
                yield Err(HarnessError::ResourceLimit(format!("Ollama stream exceeds {max_stream_bytes} bytes")));
                return;
            }
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                match parse_line(line.trim_ascii()) {
                    Ok(events) => {
                        for event in events {
                            let completed = matches!(event, OllamaStreamEvent::Completed { .. });
                            yield Ok(event);
                            if completed {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
            if buffer.len() > max_line_bytes {
                yield Err(HarnessError::ResourceLimit(format!("Ollama stream line exceeds {max_line_bytes} bytes")));
                return;
            }
        }
        if !buffer.trim_ascii().is_empty() {
            match parse_line(buffer.trim_ascii()) {
                Ok(events) => {
                    for event in events {
                        let completed = matches!(event, OllamaStreamEvent::Completed { .. });
                        yield Ok(event);
                        if completed {
                            return;
                        }
                    }
                }
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }
        yield Err(HarnessError::Provider("Ollama stream ended before a done event".into()));
    })
}

fn parse_line(line: &[u8]) -> Result<Vec<OllamaStreamEvent>, HarnessError> {
    if line.is_empty() {
        return Ok(vec![]);
    }
    let response: ChatResponse = serde_json::from_slice(line)
        .map_err(|error| HarnessError::Provider(format!("decode Ollama stream event: {error}")))?;
    let mut events = Vec::new();
    if let Some(message) = response.message.as_ref() {
        if !message.content.is_empty() {
            events.push(OllamaStreamEvent::TextDelta {
                content: message.content.clone(),
            });
        }
        events.extend(
            tool_calls(&message.tool_calls)
                .into_iter()
                .map(|call| OllamaStreamEvent::ToolCall { call }),
        );
    }
    if response.done {
        let response_usage = usage(&response);
        events.push(OllamaStreamEvent::Completed {
            model: response.model.unwrap_or_default(),
            usage: response_usage,
        });
    }
    Ok(events)
}
