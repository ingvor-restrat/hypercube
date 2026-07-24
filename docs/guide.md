# A Guided Tour of Hypercube

Hypercube is easiest to understand as two cooperating systems.

The engine turns one coherent input generation into calculated
cross-sections. Slice publishes those cross-sections as stable memory-mapped
vectors that other processes can read cheaply.

## One generation

An `Update` contains:

- a strictly increasing generation number;
- an observation time and execution mode;
- one `InputRow` per entity; and
- the node graph to evaluate.

Each row is a small field map. The same engine can therefore process
instruments, sensors, service instances, experiments, or another stable entity
set.

```rust
let row = InputRow::new("sensor-a", 1_000)
    .with_field("temperature", 21.4)
    .with_field("load", 0.73);
```

The engine rejects duplicate row keys, duplicate node identifiers, unknown
required dependencies, cycles, non-finite weights, and stale generations.
Failed updates do not advance the engine generation.

## Nodes and transforms

A field node reads one primitive field across every entity. A linear node
combines already computed nodes:

```text
temperature ──> temperature_z ─┐
                               ├──> condition
load ─────────> load_z ────────┘
```

The available cross-sectional transforms are:

| Transform | Meaning |
| --- | --- |
| `Identity` | Preserve finite input values. |
| `ZScore` | Population z-score across available entities. |
| `Rank` | Ascending one-based rank with average ranks for ties. |
| `Percentile` | Rank mapped to `[0, 1]`; one value maps to `0.5`. |
| `RankZScore` | Rank first, then z-score the ranks. |

Required inputs must exist for an entity. Optional inputs may be absent. When
linear weight normalization is enabled, the result is divided by the absolute
weights actually available to that entity.

Node declaration order is not execution order. Hypercube resolves the graph
topologically and returns values in declaration order for stable consumers.

## Snapshot coordinates

Every `CellValue` has:

```text
(generation, node, entity) -> (value, observed_at)
```

`Snapshot::slice("liquid_residual_score")` selects a node cross-section.
`Snapshot::value("liquid_residual_score", "SIM0001")` selects one cell.

The in-process snapshot is coherent across every node in that generation.

## Publishing slices

`SlicePublisher` maps selected node cross-sections onto one stable entity
layout. For each node it creates an `f64` file:

```text
layout.json
catalog.json
slices/
  price.slice
  residual_z.slice
  liquid_residual_score.slice
```

One writer surrounds a vector update with an odd/even epoch. A reader accepts
the copied vector only when the epoch was the same even value before and after
the copy. Fixed records use the same idea per slot.

This guarantees coherence within one slice. It does not make several separate
slice files globally atomic. Consumers needing a coherent decision across
multiple nodes should use the in-process `Snapshot` or add a generation
manifest and reader-pinning protocol.

## Synthetic live path

The bundled market injector is deterministic for a given seed. It simulates
log prices with shared market, sector, and stock-specific
Ornstein–Uhlenbeck states. For stock \(i\), the one-period residual is:

```text
ε_i = r_i - βM_i ΔX_M - βS_i ΔX_S(i)
```

The demo graph combines the cross-sectional residual rank with normalized log
dollar volume:

```text
score_i = z(0.75 rank-z(ε_i) + 0.25 z(log dollar_volume_i))
```

It publishes `price`, `return`, `residual_z`, `dollar_volume_z`, and
`liquid_residual_score`, then streams the snapshot through Server-Sent Events.

```bash
cargo run -p hypercube-engine --example synthetic_server
```

The browser dashboard is intentionally dependency-free. It visualizes the
residual-move score, generation history, and the current node-by-symbol grid.
The same state is available as JSON at `/api/snapshot` and as SSE at
`/api/stream`.

## Financial hello worlds

`etf` calculates constant-weight basket returns along the constituent axis,
advances each ETF fair value, and compares the simulated market price with that
fair value:

```text
rNAV_j = sum_i(w_ji r_i)
premium_bps_j = 10,000 (market_price_j - fair_value_j) / fair_value_j
```

Hypercube publishes the ETF-aligned fair values, prices, premiums, model
z-scores, and cross-sectional ranks.

```bash
cargo run -p hypercube-engine --example etf
```

`pairs` treats each pair as a stable entity. Its primitive calculation is the
cointegrating residual:

```text
y_j = log(A_j) - alpha_j - beta_j log(B_j)
```

The example standardizes \(y_j\) using its declared AR(1) process and asks
Hypercube to rank `abs(spread_z)` across the current pair set.

```bash
cargo run -p hypercube-engine --example pairs
```

The [recorded walkthrough](markup/README.md) derives both calculations and
includes bounded commands suitable for CI.
