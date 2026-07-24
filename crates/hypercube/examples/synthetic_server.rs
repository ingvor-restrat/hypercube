//! Semi-live Hypercube demonstration with an HTTP dashboard, JSON snapshot,
//! Server-Sent Events stream, and memory-mapped node slices.
//!
//! All input is deterministic synthetic data, so the complete architecture can
//! be explored without credentials, market feeds, or application services.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use hypercube::synthetic::{market_demo_nodes, OuMarketInjector};
use hypercube::{HypercubeEngine, SlicePublisher, Snapshot};

const DASHBOARD: &str = include_str!("../demo/index.html");

#[derive(Debug)]
struct Config {
    address: String,
    entities: usize,
    interval: Duration,
    slice_dir: PathBuf,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let nodes = market_demo_nodes();
    let node_ids = nodes.iter().map(|node| node.id.clone()).collect::<Vec<_>>();
    let mut injector = OuMarketInjector::new(config.entities, 0x5eed);
    let entities = injector.symbols();
    let mut engine = HypercubeEngine::new();
    let mut publisher = SlicePublisher::create_overwrite(
        &config.slice_dir,
        "synthetic-live-v1",
        &entities,
        &node_ids,
    )?;
    let initial = engine.update(injector.next_frame(now_ms()).into_update(nodes.clone()))?;
    publisher.publish(&initial)?;
    let state = Arc::new(RwLock::new(initial));

    let generator_state = Arc::clone(&state);
    let interval = config.interval;
    thread::spawn(move || loop {
        thread::sleep(interval);
        let result = (|| -> Result<Snapshot> {
            let snapshot =
                engine.update(injector.next_frame(now_ms()).into_update(nodes.clone()))?;
            publisher.publish(&snapshot)?;
            Ok(snapshot)
        })();
        match result {
            Ok(snapshot) => {
                if let Ok(mut current) = generator_state.write() {
                    *current = snapshot;
                }
            }
            Err(error) => eprintln!("synthetic update failed: {error}"),
        }
    });

    let listener = TcpListener::bind(&config.address)
        .with_context(|| format!("failed binding {}", config.address))?;
    println!("Hypercube demo: http://{}", config.address);
    println!("Memory-mapped slices: {}", config.slice_dir.display());
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, state, interval) {
                        if error.kind() != std::io::ErrorKind::BrokenPipe {
                            eprintln!("request failed: {error}");
                        }
                    }
                });
            }
            Err(error) => eprintln!("accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<RwLock<Snapshot>>,
    interval: Duration,
) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut request = [0_u8; 8_192];
    let read = stream.read(&mut request)?;
    let first_line = String::from_utf8_lossy(&request[..read])
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    let path = first_line.split_whitespace().nth(1).unwrap_or("/");
    let path = path.split('?').next().unwrap_or(path);
    match path {
        "/" | "/index.html" => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            DASHBOARD.as_bytes(),
        ),
        "/api/snapshot" => {
            let body = state
                .read()
                .ok()
                .and_then(|snapshot| serde_json::to_vec(&*snapshot).ok())
                .unwrap_or_else(|| br#"{"error":"snapshot unavailable"}"#.to_vec());
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        "/api/stream" => stream_snapshots(&mut stream, state, interval),
        "/favicon.ico" => respond(&mut stream, "204 No Content", "image/x-icon", &[]),
        _ => respond(
            &mut stream,
            "404 Not Found",
            "application/json",
            br#"{"error":"not found"}"#,
        ),
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn stream_snapshots(
    stream: &mut TcpStream,
    state: Arc<RwLock<Snapshot>>,
    interval: Duration,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n"
    )?;
    stream.flush()?;
    let mut last_generation = 0;
    loop {
        let payload = state.read().ok().and_then(|snapshot| {
            if snapshot.generation == last_generation {
                None
            } else {
                last_generation = snapshot.generation;
                serde_json::to_string(&*snapshot).ok()
            }
        });
        if let Some(payload) = payload {
            write!(stream, "event: snapshot\ndata: {payload}\n\n")?;
            stream.flush()?;
        }
        thread::sleep(interval.min(Duration::from_millis(500)));
    }
}

fn parse_args() -> Result<Config> {
    let mut address = "127.0.0.1:8080".to_owned();
    let mut entities = 32_usize;
    let mut interval_ms = 250_u64;
    let mut slice_dir = env::temp_dir().join("hypercube-demo");
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| anyhow!("{flag} requires a value"))?;
        match flag.as_str() {
            "--address" => address = value,
            "--entities" => entities = value.parse().context("invalid --entities")?,
            "--interval-ms" => interval_ms = value.parse().context("invalid --interval-ms")?,
            "--slice-dir" => slice_dir = PathBuf::from(value),
            _ => return Err(anyhow!("unknown argument {flag}")),
        }
    }
    if entities == 0 {
        return Err(anyhow!("--entities must be positive"));
    }
    if interval_ms < 25 {
        return Err(anyhow!("--interval-ms must be at least 25"));
    }
    Ok(Config {
        address,
        entities,
        interval: Duration::from_millis(interval_ms),
        slice_dir,
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}
