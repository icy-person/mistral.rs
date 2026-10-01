use futures_util::StreamExt;
use mistralrs_streaming::{
    ChatRequest, Message, ReconnectPolicy, StreamEvent, StreamingClient, StreamingError,
};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::{sleep, timeout},
};
use tokio_util::sync::CancellationToken;

fn request() -> ChatRequest {
    ChatRequest {
        model: "test".into(),
        messages: vec![Message {
            role: "user".into(),
            content: "hello".into(),
        }],
        stream: true,
        max_tokens: Some(8),
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        response_format: None,
        seed: None,
        stream_options: None,
    }
}

async fn read_headers(socket: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut request = Vec::with_capacity(1024);
    let mut buf = [0u8; 512];

    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = socket.read(&mut buf).await?;
        if count == 0 {
            break;
        }
        request.extend_from_slice(&buf[..count]);

        if request.len() > 64 * 1024 {
            break;
        }
    }

    Ok(request)
}

async fn write_sse_headers(socket: &mut TcpStream) -> std::io::Result<()> {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Cache-Control: no-cache\r\n\
Connection: close\r\n\
\r\n",
        )
        .await
}

async fn write_sse_status(socket: &mut TcpStream, status: &str) -> std::io::Result<()> {
    socket
        .write_all(
            format!(
                "HTTP/1.1 {status}\r\n\
Content-Type: text/plain\r\n\
Content-Length: 0\r\n\
Connection: close\r\n\
\r\n"
            )
            .as_bytes(),
        )
        .await
}

fn chunk(id: u32, content: &str) -> String {
    format!(
        "id: {id}\ndata: {{\"id\":\"chatcmpl-test\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{content}\"}}}}]}}\n\n"
    )
}

#[test]
fn serializes_extended_chat_request_controls() {
    let mut request = request();
    request.frequency_penalty = Some(0.25);
    request.presence_penalty = Some(0.5);
    request.stop = Some(serde_json::json!(["END", "STOP"]));
    request.tools = Some(vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    })]);
    request.tool_choice = Some(serde_json::json!({
        "type": "function",
        "function": {"name": "get_weather"}
    }));
    request.parallel_tool_calls = Some(true);
    request.response_format = Some(serde_json::json!({"type": "json_object"}));
    request.seed = Some(42);
    request.stream_options = Some(mistralrs_streaming::StreamOptions {
        include_usage: Some(true),
        include_obfuscation: Some(false),
    });

    let json = serde_json::to_value(request).unwrap();

    assert_eq!(json["frequency_penalty"], 0.25);
    assert_eq!(json["presence_penalty"], 0.5);
    assert_eq!(json["stop"], serde_json::json!(["END", "STOP"]));
    assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(json["tool_choice"]["function"]["name"], "get_weather");
    assert_eq!(json["parallel_tool_calls"], true);
    assert_eq!(json["response_format"]["type"], "json_object");
    assert_eq!(json["seed"], 42);
    assert_eq!(json["stream_options"]["include_usage"], true);
    assert_eq!(json["stream_options"]["include_obfuscation"], false);
}

#[test]
fn parses_streamed_tool_call_and_refusal_deltas() {
    let mut parser = mistralrs_streaming::SseParser::new();
    parser
        .push(
            br#"data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Baku\"}"}}]}}]}

"#,
        )
        .unwrap();

    match parser.next_event().unwrap().unwrap() {
        StreamEvent::Chunk(chunk) => {
            let tool = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
            assert_eq!(tool[0].index, 0);
            assert_eq!(tool[0].id.as_deref(), Some("call_1"));
            assert_eq!(tool[0].r#type.as_deref(), Some("function"));
            let function = tool[0].function.as_ref().unwrap();
            assert_eq!(function.name.as_deref(), Some("get_weather"));
            assert_eq!(function.arguments.as_deref(), Some(r#"{"city":"Baku"}"#));
        }
        StreamEvent::Done => panic!("unexpected done"),
    }

    parser
        .push(
            br#"data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"refusal":"not allowed"}}]}

"#,
        )
        .unwrap();

    match parser.next_event().unwrap().unwrap() {
        StreamEvent::Chunk(chunk) => {
            assert_eq!(
                chunk.choices[0].delta.refusal.as_deref(),
                Some("not allowed")
            );
        }
        StreamEvent::Done => panic!("unexpected done"),
    }
}

#[tokio::test]
async fn reconnects_with_last_event_id_without_restarting_output() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let _ = read_headers(&mut first).await.unwrap();
        write_sse_headers(&mut first).await.unwrap();
        first.write_all(chunk(1, "A").as_bytes()).await.unwrap();
        first.write_all(chunk(2, "B").as_bytes()).await.unwrap();
        first.shutdown().await.unwrap();

        let (mut second, _) = listener.accept().await.unwrap();
        let request = read_headers(&mut second).await.unwrap();
        let headers = String::from_utf8_lossy(&request).to_ascii_lowercase();
        assert!(headers.contains("last-event-id: 2\r\n"));

        write_sse_headers(&mut second).await.unwrap();
        second.write_all(chunk(3, "C").as_bytes()).await.unwrap();
        second.write_all(b"data: [DONE]\n\n").await.unwrap();
        second.shutdown().await.unwrap();
    });

    let client = StreamingClient::builder()
        .base_url(format!("http://{address}/v1"))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    let mut stream = client
        .stream_with_reconnect(
            request(),
            ReconnectPolicy {
                max_retries: 2,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(2),
            },
        )
        .await
        .unwrap();

    let mut output = String::new();
    while let Some(event) = timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        match event.unwrap() {
            StreamEvent::Chunk(chunk) => {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        output.push_str(&content);
                    }
                }
            }
            StreamEvent::Done => break,
        }
    }

    server.await.unwrap();
    assert_eq!(output, "ABC");
}

#[tokio::test]
async fn retries_retryable_http_status_after_a_resumable_event() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let _ = read_headers(&mut first).await.unwrap();
        write_sse_headers(&mut first).await.unwrap();
        first.write_all(chunk(7, "A").as_bytes()).await.unwrap();
        first.shutdown().await.unwrap();

        let (mut second, _) = listener.accept().await.unwrap();
        let request = read_headers(&mut second).await.unwrap();
        let headers = String::from_utf8_lossy(&request).to_ascii_lowercase();
        assert!(headers.contains("last-event-id: 7\r\n"));
        write_sse_status(&mut second, "503 Service Unavailable")
            .await
            .unwrap();

        let (mut third, _) = listener.accept().await.unwrap();
        let request = read_headers(&mut third).await.unwrap();
        let headers = String::from_utf8_lossy(&request).to_ascii_lowercase();
        assert!(headers.contains("last-event-id: 7\r\n"));
        write_sse_headers(&mut third).await.unwrap();
        third.write_all(chunk(8, "B").as_bytes()).await.unwrap();
        third.write_all(b"data: [DONE]\n\n").await.unwrap();
        third.shutdown().await.unwrap();
    });

    let client = StreamingClient::builder()
        .base_url(format!("http://{address}/v1"))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    let mut stream = client
        .stream_with_reconnect(
            request(),
            ReconnectPolicy {
                max_retries: 3,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(2),
            },
        )
        .await
        .unwrap();

    let mut output = String::new();
    while let Some(event) = timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        match event.unwrap() {
            StreamEvent::Chunk(chunk) => {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        output.push_str(&content);
                    }
                }
            }
            StreamEvent::Done => break,
        }
    }

    server.await.unwrap();
    assert_eq!(output, "AB");
}

#[tokio::test]
async fn cancellation_wakes_a_pending_stream_and_is_terminal() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_headers(&mut socket).await.unwrap();
        write_sse_headers(&mut socket).await.unwrap();
        socket.write_all(chunk(1, "A").as_bytes()).await.unwrap();

        sleep(Duration::from_secs(10)).await;
    });

    let client = StreamingClient::builder()
        .base_url(format!("http://{address}/v1"))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    let cancellation = CancellationToken::new();
    let mut stream = client
        .stream_with_cancellation(request(), cancellation.clone())
        .await
        .unwrap();

    match timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        Some(Ok(StreamEvent::Chunk(chunk))) => {
            assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("A"));
        }
        other => panic!("unexpected first event: {other:?}"),
    }

    let pending = timeout(Duration::from_secs(2), stream.next());
    tokio::pin!(pending);
    sleep(Duration::from_millis(20)).await;
    cancellation.cancel();

    let result = pending.await.unwrap();
    assert!(matches!(result, Some(Err(StreamingError::Cancelled))));
    assert!(stream.next().await.is_none());

    server.abort();
}
