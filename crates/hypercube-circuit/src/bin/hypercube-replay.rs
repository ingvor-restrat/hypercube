use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::process;

use hypercube::synthetic::{market_demo_nodes, OuMarketInjector};
use hypercube_circuit::{
    residual_score_trigger, CaptureSession, CircuitConfig, RecordingManifest, RecordingReader,
    ReplayRunner,
};

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => process::exit(2),
        Err(error) => {
            eprintln!("hypercube-replay: {error}");
            process::exit(1);
        }
    }
}

fn run() -> Result<bool, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("record-demo") => {
            let output = args.next().ok_or_else(usage)?;
            let generations = parse_count(args.next(), 40, "generations")?;
            let entities = parse_count(args.next(), 32, "entities")?;
            if args.next().is_some() {
                return Err(usage().into());
            }
            record_demo(Path::new(&output), generations, entities)?;
            Ok(true)
        }
        Some("verify") => {
            let input = args.next().ok_or_else(usage)?;
            if args.next().is_some() {
                return Err(usage().into());
            }
            verify(Path::new(&input))
        }
        _ => Err(usage().into()),
    }
}

fn record_demo(path: &Path, generations: usize, entities: usize) -> Result<(), Box<dyn Error>> {
    if generations == 0 {
        return Err("generations must be positive".into());
    }
    if entities == 0 {
        return Err("entities must be positive".into());
    }
    let manifest = RecordingManifest::new(
        "synthetic-factor-demo-v1",
        env!("CARGO_PKG_VERSION"),
        "ou-market-seed-335342",
        format!("synthetic-layout-{entities}"),
        vec![residual_score_trigger()?],
    )?
    .with_metadata("input_boundary", "complete_hypercube_update")?
    .with_metadata("float_contract", "bitwise")?;
    let file = BufWriter::new(File::create(path)?);
    let mut capture = CaptureSession::new(file, manifest, CircuitConfig::default())?;
    let mut injector = OuMarketInjector::new(entities, 335_342);
    let mut factor_cells = 0_usize;
    let mut transitions = 0_usize;
    for index in 0..generations {
        let observed_at_ms = 1_700_000_000_000_i64 + index as i64 * 100;
        let update = injector
            .next_frame(observed_at_ms)
            .into_update(market_demo_nodes());
        let frame = capture.process(update)?;
        factor_cells = factor_cells.saturating_add(frame.snapshot.values);
        transitions = transitions.saturating_add(frame.transitions.len());
    }
    capture.finish()?;
    println!(
        "recorded {generations} generations, {factor_cells} factor cells, and \
         {transitions} transitions to {}",
        path.display()
    );
    Ok(())
}

fn verify(path: &Path) -> Result<bool, Box<dyn Error>> {
    let recording = RecordingReader::open(path)?;
    let report = ReplayRunner::default().run(recording)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(report.is_exact())
}

fn parse_count(value: Option<String>, default: usize, name: &str) -> Result<usize, Box<dyn Error>> {
    value
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid {name}: {error}").into())
        })
        .unwrap_or(Ok(default))
}

fn usage() -> String {
    "usage:\n  hypercube-replay record-demo OUTPUT [GENERATIONS] [ENTITIES]\n  \
     hypercube-replay verify INPUT"
        .to_owned()
}
