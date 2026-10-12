# A Guided Tour of Hypercube

The engine turns one coherent input generation into calculated
cross-sections. The projection runtime gathers selected cells into new
calculated strips. Slice publishes engine cross-sections as stable
memory-mapped vectors that other processes can read cheaply.

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
The validated topology and dependency indexes are compiled once per unchanged
node declaration and reused across generations.

## Snapshot coordinates

Every `CellValue` has:

```text
(generation, node, entity) -> (value, observed_at)
```

`Snapshot::slice("liquid_residual_score")` selects a node cross-section.
`Snapshot::value("liquid_residual_score", "SIM0001")` selects one cell.

The in-process snapshot is coherent across every node in that generation.

## Indexed projections

A projection picks values from engine nodes or other projections, runs a
stateless function, and publishes an immutable contiguous `ProjectionStrip`.
Its layout is explicit and can name groups, instruments, or memberships such
as `sector/instrument`. Repeated input picks are allowed; output keys must be
unique. A projection strip lives in process and is not a `.slice` file.

The runnable example defines a semiconductor basket and a dependent residual:

```text
returns[NVDA, AMD, AVGO] -- weights[0.5, 0.3, 0.2] --> semis_return[SEMIS]
                                                           |
returns[NVDA, SPY] -----------------------------------------+
                                                           v
                 NVDA - 0.8 * SEMIS - 0.2 * SPY --> nvda_residual[NVDA]
```

```bash
cargo run -p hypercube-engine --example projections
```

The [complete example](../crates/hypercube/examples/projections.rs) starts the
engine before registering the graph. Its
[JSON config](../crates/hypercube/examples/projections.json) contains the graph
revision, selectable source-node layouts, and projection definitions. Each
definition specifies an ID, version, ordered picks, function name, parameters,
and output keys. Source namespaces explicitly distinguish an engine `node`
from a `projection`; declaration order does not determine execution order.

With an existing engine and update, registration and evaluation look like:

```rust,ignore
use std::sync::Arc;
use hypercube::projection::{FunctionRegistry, ProjectionConfig, ProjectionRuntime};

let config: ProjectionConfig = serde_json::from_str(&config_json)?;
let mut views = ProjectionRuntime::new(FunctionRegistry::with_builtins());
views.stage(config)?;
let snapshot = Arc::new(engine.update(update)?);
let frame = views.evaluate(Arc::clone(&snapshot))?;
let basket = Arc::clone(frame.strip("semis_return").unwrap());
assert_eq!(basket.keys(), &["SEMIS"]);
```

`gather@1` returns the picked values in order and takes no parameters.
`weighted_sum@1` uses one finite weight per pick and produces one value without
normalizing the weights. Register other implementations through
`ProjectionFunction`; its `validate` method checks shapes and parameters during
staging, and `calculate` returns owned values. Implementations must be
deterministic, stateless, and free of side effects. Existing function names
cannot be replaced; use a new versioned name when semantics change.

### Updating and sharing views

Call `stage` with the complete desired graph, including retained projections,
to add, replace, or remove definitions at runtime. Graph revisions increase;
changed definitions also need higher definition versions. The compiler rejects
unknown sources or keys, cycles, duplicate IDs/layout keys, and invalid function
shapes. Snapshot cells are matched by key, not by their physical position.

The next successful `evaluate` activates the pending graph and publishes all
results together. Every dependency uses the supplied snapshot's generation.
Missing or non-finite picked values, calculation errors, invalid output lengths,
and non-finite outputs fail the whole evaluation. Unpicked missing cells are
allowed. On failure, `latest()` retains the prior frame and `pending_config()`
retains the staged definition; the same generation can be retried. The engine's
own successful update is not rolled back, so retain its snapshot for that retry.
No older projection value is substituted.

To supersede a pending change, stage another complete config with a newer graph
revision and advance any affected definition versions. Successfully staged
versions cannot be repurposed, even if they never became active or were removed.
Unchanged downstream definitions can retain their versions when upstream
definitions change; the graph revision identifies the complete dependency set.

The runtime has one mutable owner and evaluates synchronously. Applications
share completed frames through their own channels or synchronization.
Readers may retain an `Arc<ProjectionStrip>` while the owner evaluates later
generations. The strip keeps its values, keys, definition version, graph
revision, and generation together; its storage is never reused beneath a reader.
The frame additionally retains the source snapshot and exact config.

### Recovering definitions

Serialize `frame.config()` with `serde_json` to capture the successfully active
definition, or `pending_config()` for a staged definition. The application owns
durable storage and chooses which revision to restore. On restart, register
compatible function implementations, deserialize the config, and call `stage`.
The example verifies identical results by evaluating the recovered graph against
the same snapshot.

Config restores definitions, not input history, output values, or earlier
generation/version guards. Historical evaluation also needs the input snapshots
and the revisions active at those generations. Circuit's current recording format
does not automatically include that projection history; see the
[replay boundary](replay.md#projection-recovery-and-replay).
Dynamic membership rules and stateful/windowed projection functions remain
outside this initial API.

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

Slice publication reuses node vectors and cached entity slots. `publish()` waits for a
durable flush for compatibility. `publish_with_durability()` can instead
return after mapped-memory visibility or after scheduling asynchronous flush;
the caller must choose that weaker postcondition deliberately.

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

The example standardizes \(y_j\) against the preceding 20 residuals using the
fixed-capacity `RollingMoments` primitive. It scores before inserting the
current observation, then asks Hypercube to rank `abs(spread_z)` across the
current pair set.

```bash
cargo run -p hypercube-engine --example pairs
```

The [recorded walkthrough](markup/README.md) derives both calculations and
includes bounded commands suitable for CI.

The [performance and stat-arb audit](performance.md) compares this window
maintenance with allocating and reusable-ring two-pass variants, and explains
why explicit SIMD and prefetching remain measured opt-ins rather than defaults.
