//! Direct, loopback-only Ollama integration for the embedded Llama Harness runtime.

#![deny(missing_docs)]

mod provider;
mod streaming;
mod wire;

pub use provider::{
    OllamaProvider, OllamaProviderBuilder, DEFAULT_OLLAMA_BASE_URL, DEFAULT_REQUEST_TIMEOUT,
};
pub use streaming::{OllamaEventStream, OllamaStreamEvent};
