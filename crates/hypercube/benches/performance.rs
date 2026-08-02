use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hypercube::{
    ExecutionMode, HypercubeEngine, InputRow, NodeSpec, PublishDurability, RollingMoments,
    SlicePublisher, Transform, Update, WeightedInput,
};

fn graph() -> Vec<NodeSpec> {
    // Declaration order is deliberately not execution order. This keeps the
    // benchmark representative of a graph that needs dependency planning.
    vec![
        NodeSpec::linear(
            "score",
            vec![
                WeightedInput::required("mix", 0.7),
                WeightedInput::required("activity_rank", 0.3),
            ],
            true,
            Transform::RankZScore,
        ),
        NodeSpec::field("left_z", "left", Transform::ZScore),
        NodeSpec::field("right_z", "right", Transform::ZScore),
        NodeSpec::field("activity_rank", "activity", Transform::RankZScore),
        NodeSpec::linear(
            "mix",
            vec![
                WeightedInput::required("left_z", 1.0),
                WeightedInput::required("right_z", -0.65),
            ],
            false,
            Transform::Identity,
        ),
    ]
}

fn update(entity_count: usize, generation: u64) -> Update {
    let rows = (0..entity_count)
        .rev()
        .map(|index| {
            let phase = index as f64 * 0.017;
            InputRow::new(format!("E{index:06}"), generation as i64 * 1_000)
                .with_field("left", phase.sin() + index as f64 * 1e-4)
                .with_field("right", phase.cos() - index as f64 * 2e-4)
                .with_field("activity", (1.0 + index as f64).ln())
        })
        .collect();
    Update {
        generation,
        observed_at_ms: generation as i64 * 1_000,
        mode: ExecutionMode::Batch,
        rows,
        nodes: graph(),
    }
}

fn benchmark_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_generation");
    for entity_count in [128_usize, 1_024] {
        group.throughput(Throughput::Elements(entity_count as u64));
        let mut engine = HypercubeEngine::new();
        let mut frame = update(entity_count, 1);
        engine.update_ref(&frame).expect("warm-up frame is valid");
        let mut generation = 1_u64;
        group.bench_with_input(
            BenchmarkId::new("stable_graph", entity_count),
            &entity_count,
            |bencher, _| {
                bencher.iter(|| {
                    generation += 1;
                    frame.generation = generation;
                    frame.observed_at_ms = generation as i64 * 1_000;
                    for row in &mut frame.rows {
                        row.observed_at_ms = frame.observed_at_ms;
                    }
                    black_box(
                        engine
                            .update_ref(black_box(&frame))
                            .expect("benchmark frame is valid"),
                    )
                });
            },
        );
    }
    group.finish();
}

fn benchmark_publisher(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_publisher");
    for entity_count in [128_usize, 1_024] {
        let mut engine = HypercubeEngine::new();
        let frame = update(entity_count, 1);
        let snapshot = engine.update_ref(&frame).expect("publisher frame is valid");
        let entities = frame
            .rows
            .iter()
            .map(|row| row.key.clone())
            .collect::<Vec<_>>();
        let nodes = frame
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().expect("temporary benchmark directory");
        let mut publisher = SlicePublisher::create(
            directory.path(),
            format!("benchmark-{entity_count}"),
            &entities,
            &nodes,
        )
        .expect("benchmark publisher is valid");
        publisher
            .publish(&snapshot)
            .expect("publisher warm-up succeeds");

        group.throughput(Throughput::Elements(snapshot.values.len() as u64));
        group.bench_function(BenchmarkId::new("durable", entity_count), |bencher| {
            bencher.iter(|| {
                publisher
                    .publish(black_box(&snapshot))
                    .expect("benchmark publication succeeds")
            });
        });
        group.bench_function(BenchmarkId::new("memory_mapped", entity_count), |bencher| {
            bencher.iter(|| {
                publisher
                    .publish_with_durability(black_box(&snapshot), PublishDurability::MemoryMapped)
                    .expect("benchmark publication succeeds")
            });
        });
    }
    group.finish();
}

fn spread_series(length: usize) -> Vec<f64> {
    (0..length)
        .map(|index| {
            let time = index as f64;
            0.004 * (time * 0.037).sin() + 0.002 * (time * 0.011).cos()
        })
        .collect()
}

fn allocating_rolling_zscores(values: &[f64], window: usize) -> f64 {
    let mut checksum = 0.0;
    for index in window..values.len() {
        let frame = values[index - window..index].to_vec();
        let mean = frame.iter().sum::<f64>() / frame.len() as f64;
        let variance = frame
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / frame.len() as f64;
        checksum += (values[index] - mean) / variance.sqrt();
    }
    checksum
}

fn ring_scan_rolling_zscores(values: &[f64], window: usize) -> f64 {
    let mut ring = values[..window].to_vec();
    let mut next = 0;
    let mut checksum = 0.0;
    for &value in &values[window..] {
        let mean = ring.iter().sum::<f64>() / ring.len() as f64;
        let variance = ring
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / ring.len() as f64;
        checksum += (value - mean) / variance.sqrt();
        ring[next] = value;
        next = (next + 1) % window;
    }
    checksum
}

fn online_rolling_zscores(values: &[f64], window: usize) -> f64 {
    let mut moments = RollingMoments::new(window).expect("benchmark window is positive");
    for &value in &values[..window] {
        moments.push(value).expect("benchmark values are finite");
    }
    let mut checksum = 0.0;
    for &value in &values[window..] {
        checksum += moments
            .z_score(value)
            .expect("benchmark values are finite")
            .expect("benchmark window has variance");
        moments.push(value).expect("benchmark values are finite");
    }
    checksum
}

fn benchmark_rolling_statarb(c: &mut Criterion) {
    const LENGTH: usize = 4_096;
    const WINDOW: usize = 32;
    let values = spread_series(LENGTH);
    let expected = allocating_rolling_zscores(&values, WINDOW);
    assert!((ring_scan_rolling_zscores(&values, WINDOW) - expected).abs() < 1e-9);
    assert!((online_rolling_zscores(&values, WINDOW) - expected).abs() < 1e-8);

    let mut group = c.benchmark_group("rolling_statarb");
    group.throughput(Throughput::Elements((LENGTH - WINDOW) as u64));
    group.bench_function("allocating_two_pass", |bencher| {
        bencher.iter(|| black_box(allocating_rolling_zscores(black_box(&values), WINDOW)));
    });
    group.bench_function("ring_scan_two_pass", |bencher| {
        bencher.iter(|| black_box(ring_scan_rolling_zscores(black_box(&values), WINDOW)));
    });
    group.bench_function("online_constant_time", |bencher| {
        bencher.iter(|| black_box(online_rolling_zscores(black_box(&values), WINDOW)));
    });
    group.finish();
}

fn full_sort_top_abs(values: &[f64], limit: usize) -> Vec<(usize, f64)> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite() && *value != 0.0)
        .collect::<Vec<_>>();
    indexed.sort_by(|(left_slot, left), (right_slot, right)| {
        right
            .abs()
            .total_cmp(&left.abs())
            .then_with(|| left_slot.cmp(right_slot))
    });
    indexed.truncate(limit);
    indexed
}

fn benchmark_top_selection(c: &mut Criterion) {
    const LENGTH: usize = 16_384;
    const LIMIT: usize = 10;
    let values = (0..LENGTH)
        .map(|index| {
            let phase = index as f64 * 0.013;
            phase.sin() * (1.0 + (index % 29) as f64)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        hypercube::slice::top_abs(&values, LIMIT),
        full_sort_top_abs(&values, LIMIT)
    );

    let mut group = c.benchmark_group("top_abs_selection");
    group.throughput(Throughput::Elements(LENGTH as u64));
    group.bench_function("full_sort", |bencher| {
        bencher.iter(|| black_box(full_sort_top_abs(black_box(&values), LIMIT)));
    });
    group.bench_function("linear_select", |bencher| {
        bencher.iter(|| black_box(hypercube::slice::top_abs(black_box(&values), LIMIT)));
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_engine,
    benchmark_publisher,
    benchmark_rolling_statarb,
    benchmark_top_selection
);
criterion_main!(benches);
