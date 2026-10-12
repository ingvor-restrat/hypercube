use std::sync::{Arc, Barrier};

use anyhow::{bail, Result};
use hypercube::projection::{
    FunctionRegistry, Pick, ProjectionConfig, ProjectionFunction, ProjectionRuntime,
    ProjectionSource,
};
use hypercube::{ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Snapshot, Transform, Update};

fn config() -> ProjectionConfig {
    serde_json::from_str(include_str!("../examples/projections.json")).unwrap()
}

fn snapshot(generation: u64) -> Arc<Snapshot> {
    Arc::new(
        HypercubeEngine::new()
            .update(Update {
                generation,
                observed_at_ms: 1000,
                mode: ExecutionMode::Live,
                rows: [("NVDA", 4.0), ("AMD", 2.0), ("AVGO", 1.0), ("SPY", 0.5)]
                    .into_iter()
                    .map(|(key, value)| {
                        InputRow::new(key, 1000).with_field("return", value * generation as f64)
                    })
                    .collect(),
                nodes: vec![NodeSpec::field("returns", "return", Transform::Identity)],
            })
            .unwrap(),
    )
}

fn runtime() -> ProjectionRuntime {
    let mut runtime = ProjectionRuntime::new(FunctionRegistry::with_builtins());
    runtime.stage(config()).unwrap();
    runtime
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}

#[test]
fn basket_residual_and_config_recovery_use_one_generation() {
    let mut runtime = runtime();
    let frame = runtime.evaluate(snapshot(7)).unwrap();
    let basket = frame.strip("semis_return").unwrap();
    close(basket.values()[0], 19.6);
    close(frame.strip("nvda_residual").unwrap().values()[0], 11.62);
    assert_eq!(basket.keys(), &["SEMIS"]);
    assert_eq!(basket.definition_version(), 1);
    assert_eq!(basket.graph_revision(), 1);
    assert_eq!(basket.generation(), 7);
    assert!(runtime.pending_config().is_none());

    let saved = serde_json::to_string(frame.config()).unwrap();
    let mut recovered = ProjectionRuntime::new(FunctionRegistry::with_builtins());
    recovered
        .stage(serde_json::from_str(&saved).unwrap())
        .unwrap();
    let replayed = recovered.evaluate(Arc::clone(frame.source())).unwrap();
    for (id, strip) in frame.strips() {
        assert_eq!(strip.values(), replayed.strip(id).unwrap().values());
    }
}

#[test]
fn revisions_activate_atomically_and_old_readers_survive() {
    let mut runtime = runtime();
    let original = runtime.evaluate(snapshot(1)).unwrap();
    let held = Arc::clone(original.strip("semis_return").unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let reader_barrier = Arc::clone(&barrier);
    let reader = std::thread::spawn(move || {
        reader_barrier.wait();
        reader_barrier.wait();
        assert_eq!(held.generation(), 1);
        assert_eq!(held.graph_revision(), 1);
        close(held.value("SEMIS").unwrap(), 2.8);
    });
    let mut next = config();
    next.revision = 2;
    next.projections[1].version = 2;
    next.projections[1].parameters = vec![1.0, 0.0, 0.0];
    runtime.stage(next).unwrap();
    assert_eq!(runtime.latest().unwrap().config().revision, 1);
    barrier.wait();
    let updated = runtime.evaluate(snapshot(2)).unwrap();
    close(updated.strip("semis_return").unwrap().values()[0], 8.0);
    close(updated.strip("nvda_residual").unwrap().values()[0], 1.4);
    assert_eq!(updated.config().revision, 2);
    assert_eq!(
        updated.strip("semis_return").unwrap().definition_version(),
        2
    );
    barrier.wait();
    reader.join().unwrap();
    assert_eq!(original.config().revision, 1);
}

#[test]
fn missing_input_does_not_publish_or_activate_and_same_generation_can_retry() {
    let mut runtime = runtime();
    let old = runtime.evaluate(snapshot(1)).unwrap();
    let mut next = config();
    next.revision = 2;
    runtime.stage(next).unwrap();
    let mut missing = (*snapshot(2)).clone();
    missing.values.retain(|cell| cell.key != "AMD");
    assert!(runtime.evaluate(Arc::new(missing)).is_err());
    assert!(Arc::ptr_eq(runtime.latest().unwrap(), &old));
    assert_eq!(runtime.pending_config().unwrap().revision, 2);
    let frame = runtime.evaluate(snapshot(2)).unwrap();
    assert_eq!(frame.config().revision, 2);
    assert!(runtime.evaluate(snapshot(2)).is_err());
    assert!(Arc::ptr_eq(runtime.latest().unwrap(), &frame));
}

#[test]
fn picks_follow_keys_not_snapshot_order_and_allow_repeated_memberships() {
    let mut config = config();
    config.projections.truncate(1);
    let spec = &mut config.projections[0];
    spec.function = "gather@1".into();
    spec.parameters.clear();
    spec.picks = ["AMD", "NVDA", "AMD"]
        .into_iter()
        .map(|key| Pick {
            source: ProjectionSource::Node("returns".into()),
            key: key.into(),
        })
        .collect();
    spec.output_keys = ["first/AMD", "first/NVDA", "second/AMD"]
        .map(String::from)
        .to_vec();
    let mut runtime = ProjectionRuntime::new(FunctionRegistry::with_builtins());
    runtime.stage(config).unwrap();
    let mut input = (*snapshot(1)).clone();
    input.values.reverse();
    input.values.retain(|cell| cell.key != "AVGO"); // Unpicked missing cells are fine.
    let frame = runtime.evaluate(Arc::new(input)).unwrap();
    assert_eq!(
        frame.strip("nvda_residual").unwrap().values(),
        &[2.0, 4.0, 2.0]
    );
}

#[test]
fn invalid_graphs_leave_the_pending_graph_untouched() {
    let mut cases = Vec::new();
    let mut unknown = config();
    unknown.projections[0].picks[1].source = ProjectionSource::Projection("absent".into());
    cases.push(unknown);
    let mut cycle = config();
    cycle.projections[1].picks[0] = Pick {
        source: ProjectionSource::Projection("nvda_residual".into()),
        key: "NVDA".into(),
    };
    cases.push(cycle);
    let mut invalid_key = config();
    invalid_key.projections[0].picks[0].key = "missing".into();
    cases.push(invalid_key);
    let mut duplicate = config();
    duplicate.projections.push(duplicate.projections[0].clone());
    cases.push(duplicate);
    let mut duplicate_layout = config();
    duplicate_layout
        .node_layouts
        .get_mut("returns")
        .unwrap()
        .push("NVDA".into());
    cases.push(duplicate_layout);
    let mut missing_function = config();
    missing_function.projections[0].function = "uninstalled@1".into();
    cases.push(missing_function);
    let mut shape = config();
    shape.projections[0].parameters.pop();
    cases.push(shape);
    let mut nonfinite = config();
    nonfinite.projections[0].parameters[0] = f64::NAN;
    cases.push(nonfinite);
    for mut invalid in cases {
        let mut runtime = runtime();
        invalid.revision = 2;
        for spec in &mut invalid.projections {
            spec.version = 2;
        }
        assert!(runtime.stage(invalid).is_err());
        assert_eq!(runtime.pending_config(), Some(&config()));
        assert_eq!(runtime.evaluate(snapshot(1)).unwrap().config().revision, 1);
    }
}

#[test]
fn versions_cannot_be_reused_even_after_removal() {
    let mut runtime = runtime();
    let mut changed = config();
    changed.revision = 2;
    changed.projections[1].parameters[0] = 0.9;
    assert!(runtime.stage(changed.clone()).is_err());
    let mut empty = config();
    empty.revision = 2;
    empty.projections.clear();
    runtime.stage(empty).unwrap();
    changed.revision = 3;
    assert!(runtime.stage(changed.clone()).is_err());
    changed.projections[1].version = 2;
    runtime.stage(changed).unwrap();
    assert!(runtime.stage(config()).is_err());
}

struct BadFunction(u8);

impl ProjectionFunction for BadFunction {
    fn validate(&self, _: usize, _: &[f64], _: usize) -> Result<()> {
        Ok(())
    }
    fn calculate(&self, _: &[f64], _: &[f64]) -> Result<Vec<f64>> {
        match self.0 {
            0 => Ok(vec![]),
            1 => Ok(vec![f64::INFINITY]),
            _ => bail!("calculation failed"),
        }
    }
}

#[test]
fn function_errors_and_invalid_outputs_never_publish_partial_frames() {
    for failure in 0..3 {
        let mut runtime = runtime();
        let original = runtime.evaluate(snapshot(1)).unwrap();
        runtime
            .register_function("bad@1", BadFunction(failure))
            .unwrap();
        assert!(runtime.register_function("bad@1", BadFunction(0)).is_err());
        let mut next = config();
        next.revision = 2;
        next.projections[0].version = 2;
        next.projections[0].function = "bad@1".into();
        runtime.stage(next).unwrap();
        assert!(runtime.evaluate(snapshot(2)).is_err());
        assert!(Arc::ptr_eq(runtime.latest().unwrap(), &original));
        assert_eq!(runtime.pending_config().unwrap().revision, 2);
    }
}

#[test]
fn malformed_snapshot_and_overflow_are_rejected() {
    let mut runtime = runtime();
    let mut duplicate = (*snapshot(1)).clone();
    duplicate.values.push(duplicate.values[0].clone());
    assert!(runtime.evaluate(Arc::new(duplicate)).is_err());
    let mut nonfinite = (*snapshot(1)).clone();
    nonfinite.values[0].value = f64::NAN;
    assert!(runtime.evaluate(Arc::new(nonfinite)).is_err());
    let mut overflow = config();
    overflow.revision = 2;
    overflow.projections[1].version = 2;
    overflow.projections[1].parameters = vec![f64::MAX; 3];
    runtime.stage(overflow).unwrap();
    assert!(runtime.evaluate(snapshot(1)).is_err());
    assert!(runtime.latest().is_none());
}
