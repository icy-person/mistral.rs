use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mistralrs_streaming::SseParser;

const EVENT: &[u8] =
    br#"data: {"id":"chatcmpl-bench","choices":[{"index":0,"delta":{"content":"hello world"}}]}

"#;

const CRLF_EVENT: &[u8] =
    br#"data: {"id":"chatcmpl-bench","choices":[{"index":0,"delta":{"content":"hello world"}}]}

"#;

fn bench_single_line(c: &mut Criterion) {
    c.bench_function("sse_single_line", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            parser.push(black_box(EVENT)).unwrap();
            black_box(parser.next_event()).unwrap().unwrap();
        });
    });
}

fn bench_crlf(c: &mut Criterion) {
    c.bench_function("sse_crlf", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            parser.push(black_box(CRLF_EVENT)).unwrap();
            black_box(parser.next_event()).unwrap().unwrap();
        });
    });
}

fn bench_incremental(c: &mut Criterion) {
    c.bench_function("sse_incremental_byte_chunks", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            for byte in EVENT {
                parser.push(std::slice::from_ref(byte)).unwrap();
                black_box(parser.next_event());
            }
        });
    });
}

criterion_group!(sse_parser, bench_single_line, bench_crlf, bench_incremental);
criterion_main!(sse_parser);
