#[tokio::main]
async fn main() {
    if let Err(error) = llama_harness_runtime::serve_stdio().await {
        eprintln!("llama-harness-runtime: {error}");
        std::process::exit(1);
    }
}
