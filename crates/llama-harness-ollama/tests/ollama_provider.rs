use futures_util::StreamExt;
use llama_harness_core::{
    GenerationOptions, HarnessError, Message, MessageRole, ModelProvider, ModelRequest, ToolCall,
    ToolDefinition, ToolRisk,
};
use llama_harness_ollama::{OllamaProvider, OllamaStreamEvent};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
    time::sleep,
};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
struct CapturedRequest {
    path: String,
    body: Vec<u8>,
}

fn request(cancellation: CancellationToken) -> ModelRequest {
    ModelRequest {
        model: "qwen3:8b".into(),
        messages: vec![
            Message::system("Be concise."),
            Message::user("Inspect tasks"),
        ],
        tools: vec![],
        generation: GenerationOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_output_tokens: Some(64),
        },
        metadata: Default::default(),
        cancellation,
    }
}

fn tool() -> ToolDefinition {
    ToolDefinition {
        id: "list_tasks".into(),
        name: "List tasks".into(),
        description: "List the current tasks".into(),
        arguments_schema: json!({"type":"object","additionalProperties":false}),
        risk: ToolRisk::Low,
        idempotent: true,
        read_only: true,
    }
}

fn provider(base_url: &str) -> OllamaProvider {
    OllamaProvider::builder()
        .base_url(base_url)
        .build()
        .unwrap()
}

fn json_response(status: u16, body: Value) -> Vec<u8> {
    let body = body.to_string();
    format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

async fn server(responses: Vec<Vec<u8>>) -> (String, JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for response in responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            requests.push(read_request(&mut socket).await);
            socket.write_all(&response).await.unwrap();
            socket.shutdown().await.unwrap();
        }
        requests
    });
    (format!("http://{address}"), task)
}

async fn read_request(socket: &mut TcpStream) -> CapturedRequest {
    let mut bytes = Vec::new();
    let mut scratch = [0_u8; 1024];
    let header_end = loop {
        let count = socket.read(&mut scratch).await.unwrap();
        assert_ne!(count, 0, "request closed before headers");
        bytes.extend_from_slice(&scratch[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    let path = header.split_whitespace().nth(1).unwrap().to_owned();
    let content_length = header
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
        })
        .unwrap_or_default();
    while bytes.len() < header_end + content_length {
        let count = socket.read(&mut scratch).await.unwrap();
        assert_ne!(count, 0, "request closed before body");
        bytes.extend_from_slice(&scratch[..count]);
    }
    CapturedRequest {
        path,
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

#[tokio::test]
async fn health_and_model_inventory_use_direct_ollama_endpoints() {
    let (base_url, task) = server(vec![
        json_response(200, json!({"version":"0.12.9"})),
        json_response(
            200,
            json!({"models":[{"name":"qwen3:8b"},{"name":"gemma3:4b"}]}),
        ),
    ])
    .await;
    let provider = provider(&base_url);

    let health = provider.health().await.unwrap();
    let models = provider.list_models().await.unwrap();

    assert!(health.healthy);
    assert_eq!(health.detail.as_deref(), Some("Ollama 0.12.9"));
    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        ["qwen3:8b", "gemma3:4b"]
    );
    assert!(models[0].capabilities.supports_tools);
    let requests = task.await.unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|request| request.path.as_str())
            .collect::<Vec<_>>(),
        ["/api/version", "/api/tags"]
    );
}

#[tokio::test]
async fn chat_maps_generation_messages_tools_tool_calls_and_usage() {
    let (base_url, task) = server(vec![json_response(
        200,
        json!({
            "model":"qwen3:8b",
            "done":true,
            "message":{"role":"assistant","content":"I will list them.","tool_calls":[{"function":{"name":"list_tasks","arguments":{}}}]},
            "prompt_eval_count":12,
            "eval_count":5
        }),
    )])
    .await;
    let provider = OllamaProvider::builder()
        .base_url(&base_url)
        .keep_alive("5m")
        .build()
        .unwrap();
    let mut model_request = request(CancellationToken::new());
    model_request.tools = vec![tool()];

    let response = provider.complete(model_request).await.unwrap();

    assert_eq!(response.final_output.as_deref(), Some("I will list them."));
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].id, "ollama-0");
    assert_eq!(response.tool_calls[0].tool_id, "list_tasks");
    assert_eq!(response.tool_calls[0].arguments_json, "{}");
    assert_eq!(response.usage.input_tokens, 12);
    assert_eq!(response.usage.output_tokens, 5);

    let requests = task.await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(requests[0].path, "/api/chat");
    assert_eq!(body["stream"], false);
    assert_eq!(body["keep_alive"], "5m");
    assert_eq!(
        body["options"],
        json!({"temperature":0.2,"top_p":0.9,"num_predict":64})
    );
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["tools"][0]["function"]["name"], "list_tasks");
}

#[tokio::test]
async fn tool_only_response_continues_the_core_tool_loop() {
    let (base_url, task) = server(vec![json_response(
        200,
        json!({
            "model":"qwen3:8b",
            "done":true,
            "message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"list_tasks","arguments":{}}}]}
        }),
    )])
    .await;
    let provider = provider(&base_url);

    let response = provider
        .complete(request(CancellationToken::new()))
        .await
        .unwrap();

    assert_eq!(response.final_output, None);
    assert_eq!(response.tool_calls.len(), 1);
    task.await.unwrap();
}

#[tokio::test]
async fn malformed_tool_arguments_are_preserved_for_model_recovery() {
    let (base_url, task) = server(vec![json_response(
        200,
        json!({"done":true,"message":{"content":"repaired"}}),
    )])
    .await;
    let provider = provider(&base_url);
    let mut model_request = request(CancellationToken::new());
    model_request.messages.push(Message {
        role: MessageRole::Assistant,
        content: String::new(),
        tool_call_id: None,
        tool_calls: vec![ToolCall {
            id: "original-call".into(),
            tool_id: "list_tasks".into(),
            arguments_json: "{not valid JSON".into(),
        }],
    });

    provider.complete(model_request).await.unwrap();

    let request: Value = serde_json::from_slice(&task.await.unwrap()[0].body).unwrap();
    assert_eq!(
        request["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{not valid JSON"
    );
}

#[tokio::test]
async fn provider_errors_are_classified_and_bounded() {
    let (base_url, task) = server(vec![
        json_response(404, json!({"error":"model not found"})),
        json_response(503, json!({"error":"temporarily unavailable"})),
        json_response(
            200,
            json!({"done":true,"message":{"content":"this response is intentionally too long"}}),
        ),
    ])
    .await;
    let normal = provider(&base_url);
    assert!(
        matches!(normal.complete(request(CancellationToken::new())).await, Err(HarnessError::Provider(message)) if message.contains("model not found"))
    );
    assert!(
        matches!(normal.complete(request(CancellationToken::new())).await, Err(HarnessError::RetryableProvider(message)) if message.contains("temporarily unavailable"))
    );
    let small = OllamaProvider::builder()
        .base_url(base_url)
        .max_response_bytes(20)
        .build()
        .unwrap();
    assert!(matches!(
        small.complete(request(CancellationToken::new())).await,
        Err(HarnessError::ResourceLimit(_))
    ));
    task.await.unwrap();

    let (base_url, task) = server(vec![
        b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nnot-json".to_vec(),
    ])
    .await;
    assert!(matches!(
        provider(&base_url)
            .complete(request(CancellationToken::new()))
            .await,
        Err(HarnessError::Provider(message)) if message.contains("decode")
    ));
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_handles_fragmented_ndjson_tool_calls_and_completion() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        let body = concat!(
            "{\"model\":\"qwen3:8b\",\"message\":{\"content\":\"hel\"},\"done\":false}\n",
            "{\"model\":\"qwen3:8b\",\"message\":{\"content\":\"lo\",\"tool_calls\":[{\"function\":{\"name\":\"list_tasks\",\"arguments\":{}}}]},\"done\":false}\n",
            "{\"model\":\"qwen3:8b\",\"done\":true,\"prompt_eval_count\":3,\"eval_count\":2}\n"
        );
        socket
            .write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n{}", &body[..31]).as_bytes())
            .await
            .unwrap();
        sleep(Duration::from_millis(5)).await;
        socket.write_all(&body.as_bytes()[31..]).await.unwrap();
        socket.shutdown().await.unwrap();
    });
    let provider = OllamaProvider::builder()
        .base_url(base_url)
        .max_stream_line_bytes(200)
        .build()
        .unwrap();

    let events = provider
        .stream_chat(request(CancellationToken::new()))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 4);
    assert!(matches!(&events[0], Ok(OllamaStreamEvent::TextDelta { content }) if content == "hel"));
    assert!(matches!(&events[1], Ok(OllamaStreamEvent::TextDelta { content }) if content == "lo"));
    assert!(
        matches!(&events[2], Ok(OllamaStreamEvent::ToolCall { call }) if call.tool_id == "list_tasks" && call.arguments_json == "{}")
    );
    assert!(
        matches!(&events[3], Ok(OllamaStreamEvent::Completed { model, usage }) if model == "qwen3:8b" && usage.input_tokens == 3 && usage.output_tokens == 2)
    );
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_rejects_unbounded_or_incomplete_ndjson() {
    let (base_url, task) = server(vec![json_response(
        200,
        json!({"model":"qwen3:8b","message":{"content":"missing done"},"done":false}),
    )])
    .await;
    let events = provider(&base_url)
        .stream_chat(request(CancellationToken::new()))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(events.last(), Some(Err(HarnessError::Provider(message))) if message.contains("before a done"))
    );
    task.await.unwrap();

    let (base_url, task) = server(vec![json_response(
        200,
        json!({"model":"qwen3:8b","message":{"content":"this event is too large"},"done":true}),
    )])
    .await;
    let small = OllamaProvider::builder()
        .base_url(base_url)
        .max_stream_bytes(8)
        .build()
        .unwrap();
    let events = small
        .stream_chat(request(CancellationToken::new()))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.as_slice(),
        [Err(HarnessError::ResourceLimit(_))]
    ));
    task.await.unwrap();

    let (base_url, task) = server(vec![
        b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot-json\n".to_vec(),
    ])
    .await;
    let events = provider(&base_url)
        .stream_chat(request(CancellationToken::new()))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.as_slice(),
        [Err(HarnessError::Provider(message))] if message.contains("decode")
    ));
    task.await.unwrap();
}

#[tokio::test]
async fn cancellation_timeout_and_loopback_controls_fail_safely() {
    assert!(matches!(
        OllamaProvider::builder()
            .base_url("http://192.168.1.10:11434")
            .build(),
        Err(HarnessError::InvalidRequest(_))
    ));
    assert!(matches!(
        OllamaProvider::builder()
            .base_url("https://example.com")
            .build(),
        Err(HarnessError::InvalidRequest(_))
    ));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        sleep(Duration::from_millis(100)).await;
        let _ = socket
            .write_all(&json_response(
                200,
                json!({"done":true,"message":{"content":"late"}}),
            ))
            .await;
    });
    let timed = OllamaProvider::builder()
        .base_url(&base_url)
        .request_timeout(Duration::from_millis(10))
        .build()
        .unwrap();
    assert!(matches!(
        timed.complete(request(CancellationToken::new())).await,
        Err(HarnessError::RetryableProvider(_))
    ));
    task.abort();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        sleep(Duration::from_millis(100)).await;
        let _ = socket
            .write_all(&json_response(
                200,
                json!({"done":true,"message":{"content":"late"}}),
            ))
            .await;
    });
    let cancellation = CancellationToken::new();
    let cancelled_provider = provider(&base_url);
    let pending = cancelled_provider.complete(request(cancellation.clone()));
    tokio::pin!(pending);
    tokio::select! {
        _ = sleep(Duration::from_millis(5)) => cancellation.cancel(),
        result = &mut pending => panic!("request unexpectedly completed: {result:?}"),
    }
    assert!(matches!(pending.await, Err(HarnessError::Cancelled)));
    task.abort();
}

#[tokio::test]
async fn real_ollama_smoke_is_opt_in() {
    if std::env::var("LLAMA_HARNESS_TEST_OLLAMA").ok().as_deref() != Some("1") {
        return;
    }
    let provider = OllamaProvider::new().unwrap();
    assert!(
        provider.health().await.unwrap().healthy,
        "Ollama should be running"
    );
    let models = provider.list_models().await.unwrap();
    let Some(model) = models.first() else {
        return;
    };
    let mut smoke_request = request(CancellationToken::new());
    smoke_request.model = model.id.clone();
    smoke_request.messages = vec![Message::user("Reply with the word: smoke")];
    smoke_request.generation.max_output_tokens = Some(8);
    let response = provider.complete(smoke_request).await.unwrap();
    assert!(response.final_output.is_some() || !response.tool_calls.is_empty());
}
