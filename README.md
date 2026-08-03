# Hypercube

Hypercube is an in-memory multidimensional analytics engine for low-latency
computation and linear algebra over changing data.

Hypercube is part of the open-source analytical architecture behind
strategynet.ai. Read the
[Hypercube architecture overview](https://strategynet.ai/insights/hypercube-open-source-live-analytics).

The repository has two core parts:

- **Slice files** are typed, file-backed, memory-mapped vectors aligned to a
  stable entity layout.
- **The Hypercube engine** evaluates fields and derived nodes over complete
  entity generations.

The optional **Circuit** crate adds an ordered Disruptor stage around complete
Hypercube generations. It hosts stateful callbacks, records their inputs and
outputs, and can replay a run through fresh engine and callback state.

Rows can represent financial instruments, sensors, services, experiments, or
any other stable entity set. Hypercube does not include a database, message
transport, vendor feed, or effectful action system.

The publishable Cargo package is named `hypercube-engine`; its Rust library
name remains `hypercube`.

## See it live

### Hypercube: ETF Arbitrage

The `etf` monitor values synthetic ETFs from their constituent returns and
ranks the premiums and discounts to basket value:

```bash
cargo run -p hypercube-engine --example etf
```

[![ETF arbitrage terminal recording](docs/markup/etf.gif)](docs/markup/README.md#hypercube-etf-arbitrage)

### Hypercube: Pairs

The `pairs` monitor generates cointegrated price paths, standardizes each
log-price residual against its AR(1) model, and ranks the live dislocations:

```bash
cargo run -p hypercube-engine --example pairs
```

[![Pairs terminal recording](docs/markup/pairs.gif)](docs/markup/README.md#hypercube-pairs)

The [recorded walkthrough](docs/markup/README.md) gives both calculations,
assumptions, and bounded commands. Press `q` to leave either monitor.

A browser example publishes a changing financial cross-section into
memory-mapped slices and streams it over HTTP:

```bash
cargo run -p hypercube-engine --example synthetic_server
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080). The example:

1. generates a deterministic correlated cross-section every 250 ms;
2. removes the simulated market and sector moves from each stock return;
3. combines the residual rank with normalized log dollar volume;
4. publishes every output as an entity-aligned `.slice` file; and
5. streams the resulting cube to a dependency-free browser visualization.

Useful options:

```bash
cargo run -p hypercube-engine --example synthetic_server -- \
  --address 127.0.0.1:9090 \
  --entities 64 \
  --interval-ms 100 \
  --slice-dir /tmp/hypercube-demo
```

The demo exposes `GET /api/snapshot` and an SSE stream at `GET /api/stream`.
Its memory-mapped layout, catalog, and vectors are written beneath the selected
slice directory.

### Record and replay

Run the Hypercube callback and replay terminal demo:

```bash
cargo run -p hypercube-circuit --example circuit
```

[![Hypercube callback and replay recording](docs/markup/circuit-replay.gif)](docs/markup/README.md#hypercube-callbacks-triggers-and-replay)

Record a deterministic factor calculation plus its persistent threshold state
and transitions:

```bash
cargo run -p hypercube-circuit --bin hypercube-replay -- \
  record-demo /tmp/hypercube-factor.jsonl 40 32
```

Recalculate it from a fresh engine and compare every generation:

```bash
cargo run -p hypercube-circuit --bin hypercube-replay -- \
  verify /tmp/hypercube-factor.jsonl
```

The [replay guide](docs/replay.md) defines the exactness contract, stateful
processing model, replay isolation rule, and the intended Aeron Archive
mapping.

## Library API

```rust
use hypercube::{
    ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Transform, Update,
    WeightedInput,
};

let rows = vec![
    InputRow::new("A", 1_000).with_field("price", 10.0),
    InputRow::new("B", 1_000).with_field("price", 12.0),
];
let nodes = vec![
    NodeSpec::field("price_rank", "price", Transform::RankZScore),
    NodeSpec::linear(
        "score",
        vec![WeightedInput::required("price_rank", 1.0)],
        true,
        Transform::Identity,
    ),
];
let snapshot = HypercubeEngine::new().update(Update {
    generation: 1,
    observed_at_ms: 1_000,
    mode: ExecutionMode::Live,
    rows,
    nodes,
})?;

assert!(snapshot.value("score", "B").unwrap() > 0.0);
# Ok::<(), hypercube::CubeError>(())
```

Node order is not execution order. Hypercube resolves dependencies
topologically, rejects missing required inputs and cycles, handles missing
entity values explicitly, and produces a deterministic snapshot for each
strictly increasing generation. An unchanged node declaration reuses its
compiled execution plan across generations; `graph_compilations()` exposes
unexpected topology churn.

## Slice API

```rust
use hypercube::slice::{F64SliceReader, F64SliceWriter, LayoutRegistry};

let entities = vec!["A".to_owned(), "B".to_owned()];
let layout = LayoutRegistry::from_entities("example-v1", "example", 2, &entities)?;
let mut writer = F64SliceWriter::create("/tmp/example.slice", &layout, true)?;
writer.update_vector(|values| values.copy_from_slice(&[1.0, 2.0]))?;
writer.flush()?;

let reader = F64SliceReader::open("/tmp/example.slice")?;
assert_eq!(reader.snapshot_vec()?, vec![1.0, 2.0]);
# Ok::<(), anyhow::Error>(())
```

Slice also provides fixed-record quote, trade, and quote-at-trade payloads for
the first financial adapter, plus layout/catalog validation, guarded point
reads, stable vector snapshots, sums, dot products, and bounded top-absolute
selection.

`SlicePublisher::publish()` retains its synchronous durability barrier.
Low-latency current-state views can explicitly choose memory-mapped visibility
or asynchronous flush through `PublishDurability`; those weaker policies do
not claim disk durability.

## Performance and stat-arb audit

The engine includes a fixed-capacity `RollingMoments` primitive for explicit
state owners such as simulations and circuit callbacks. The pairs monitor uses
it to score each cointegrating residual against the preceding 20 observations,
without using the current residual to estimate its own mean and variance.

The [performance and statistical-arbitrage audit](docs/performance.md)
cross-checks Hypercube against the Imperial HFT design-pattern and pairs
benchmarks, records which techniques transferred, and publishes reproducible
Criterion results. Run the committed harness with:

```bash
cargo bench -p hypercube-engine --bench performance
```

On the documented development host, execution-plan reuse reduced stable-graph
latency by 8.55–11.59%, indexed publication reduced the 1,024-entity durable
case by 62.08%, constant-time rolling moments were 3.51× faster than an
allocating two-pass window, and bounded top-10 selection was 6.20× faster than
a full sort. These are scoped microbenchmarks, not production latency claims.

The separate [leakage-controlled stat-arb ML study](experiments/statarb_ml/README.md)
compares rolling-z selection with LightGBM and CatBoost L2/RMSE, Huber,
quantile, and calibrated three-class triple-barrier policies. Across five
deterministic seeds, the stationary control favored the simple rule on
average; in a synthetic observable regime-and-heavy-tail positive control,
CatBoost Huber improved the descriptive trade-return t-statistic by 0.664 and
mean normalized net return by 48.176, winning all five seed comparisons. The
study is an optional Python research harness, not a Rust runtime dependency or
a claim of market profitability.

## Repository layout

```text
crates/hypercube        `hypercube-engine` package, demo, and dashboard
crates/hypercube-slice  memory-mapped vector format and readers/writers
crates/hypercube-circuit stateful Disruptor callbacks and record/replay
docs                    guides, terminal recording, papers, and format notes
experiments             optional leakage-controlled research harnesses
```

The initial extraction intentionally excludes private service configuration,
data-vendor integrations, database writers, hard-coded factor registries, and
application-specific rollups. See the
[architecture](docs/architecture.md)
and
[slice format](docs/slice-format.md)
notes.

## Status

Slice format version 1 is little-endian and designed for one writer with many
readers. Recording format version 1 captures complete Hypercube updates and
deterministic threshold-trigger state and transitions. Rolling statistics are
an explicit reusable state primitive, not an implicit graph cell. Multi-slice
atomic generations, cross-language ABI fixtures, replication, raw-feed
generation assembly, Aeron Archive adapters, checkpoints, and general stateful
function cells remain future work.

Continue with the
[guided tour](docs/guide.md),
the
[live-example walkthrough](docs/markup/README.md),
the
[architecture boundary](docs/architecture.md),
the
[performance and stat-arb audit](docs/performance.md),
the
[stat-arb ML study](experiments/statarb_ml/README.md),
the
[slice format](docs/slice-format.md),
the
[record/replay contract](docs/replay.md),
and the
[reproducible results](docs/results.md).
The [long-form papers](docs/latex/README.md) cover Hypercube foundations,
callbacks, triggers, and replay.

Licensed under Apache-2.0.
