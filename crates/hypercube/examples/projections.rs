//! Register a basket and dependent residual after the engine has started.

use std::sync::Arc;

use anyhow::Result;
use hypercube::projection::{FunctionRegistry, ProjectionConfig, ProjectionRuntime};
use hypercube::{ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Transform, Update};

fn update(generation: u64) -> Update {
    Update {
        generation,
        observed_at_ms: generation as i64 * 1_000,
        mode: ExecutionMode::Live,
        rows: [
            ("NVDA", 0.04),
            ("AMD", 0.02),
            ("AVGO", 0.01),
            ("SPY", 0.005),
        ]
        .into_iter()
        .map(|(key, value)| {
            InputRow::new(key, generation as i64 * 1_000)
                .with_field("return", value * generation as f64)
        })
        .collect(),
        nodes: vec![NodeSpec::field("returns", "return", Transform::Identity)],
    }
}

fn main() -> Result<()> {
    let mut engine = HypercubeEngine::new();
    engine.update(update(1))?;

    // The engine is already running when the new graph is registered.
    let config: ProjectionConfig = serde_json::from_str(include_str!("projections.json"))?;
    let mut projections = ProjectionRuntime::new(FunctionRegistry::with_builtins());
    projections.stage(config)?;
    let frame = projections.evaluate(Arc::new(engine.update(update(2))?))?;
    for (id, strip) in frame.strips() {
        println!(
            "generation={} revision={} {id}: {:?} = {:?}",
            strip.generation(),
            strip.graph_revision(),
            strip.keys(),
            strip.values()
        );
    }

    // Config persistence is owned by the application. Re-register the same
    // function implementations on restart, then load the saved definitions.
    let saved = serde_json::to_string_pretty(frame.config())?;
    let mut recovered = ProjectionRuntime::new(FunctionRegistry::with_builtins());
    recovered.stage(serde_json::from_str(&saved)?)?;
    let replayed = recovered.evaluate(Arc::clone(frame.source()))?;
    for (id, strip) in frame.strips() {
        assert_eq!(strip.values(), replayed.strip(id).unwrap().values());
    }

    // A reader can retain generation 2 while the owner publishes generation 3.
    let held = Arc::clone(frame.strip("nvda_residual").unwrap());
    projections.evaluate(Arc::new(engine.update(update(3))?))?;
    assert_eq!(held.generation(), 2);
    println!("Recovered identical results from config; retained reader still has generation 2.");
    Ok(())
}
