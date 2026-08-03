# Reproducible Results

These results were recorded from Hypercube `main` on 2 August 2026. Functional
checks validate behavior and packaging. The separate Criterion section is a
scoped development-host benchmark, not a production latency claim.

## Test suite

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo +1.81.0 test --workspace --all-targets
```

Result:

| Suite | Tests | Result |
| --- | ---: | --- |
| engine, publisher, and synthetic injectors | 7 | passed |
| fixed-capacity rolling moments | 6 | passed |
| ETF and pairs example invariants | 2 | passed |
| slice layout, catalog, mmap, records, and vector algebra | 8 | passed |
| circuit, triggers, recording, digest, and replay | 14 | passed |
| **Total** | **37** | **passed** |

The suite includes dependency-cycle rejection, stale-generation rejection,
stable-graph plan reuse, tie-aware ranking, deterministic generic and OU market
injection, long-only ETF basket weights, reconstruction of the declared pair
residual, rolling-moment agreement with corrected two-pass windows,
no-look-ahead scoring, large-offset and sub-epsilon variance retention, aligned
memory-only publication, deterministic bounded top-k selection, layout
mismatch rejection, live heartbeat observation,
quote/trade/TAQ records, guarded dot products, ordered Disruptor processing,
persistent and hysteretic trigger transitions, missing-data invalidation,
recording validation, non-finite input rejection, exact floating-point JSON
round trips, semantic divergence detection, and fresh-state replay.

The complete all-target suite, including the benchmark smoke harness, also
passes with `rustc 1.81.0`; benchmark-only dependency versions are pinned so
the workspace's declared minimum remains reproducible.

## Documentation contract

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All three crates build their complete public API documentation without
warnings. Missing public documentation is enabled at the crate level, and CI
promotes warnings to errors for the documentation build.

## Performance benchmark

The committed statistics-driven harness is:

```bash
cargo bench -p hypercube-engine --bench performance
```

Criterion 0.7 preserves the workspace's Rust 1.81 minimum. The following are
central estimates from 30 samples after a one-second warmup and a three-second
measurement target. Before/after lanes used the same benchmark source and
Criterion version.

| Benchmark | Before / reference | Current | Result |
| --- | ---: | ---: | ---: |
| stable graph, 128 entities | 106.18 µs | 96.99 µs | 8.55% lower latency |
| stable graph, 1,024 entities | 998.31 µs | 888.30 µs | 11.59% lower latency |
| durable five-slice publish, 128 entities | 3.308 ms | 3.314 ms | no significant change |
| durable five-slice publish, 1,024 entities | 9.071 ms | 3.494 ms | 62.08% lower latency |
| memory-mapped five-slice publish, 1,024 entities | — | 109.04 µs | 46.96 million cells/s |
| rolling z-score, allocating two-pass | 156.43 µs | 44.53 µs online | 3.51× |
| top 10 of 16,384 values, full sort | 296.92 µs | 47.92 µs select | 6.20× |

Host: AMD Ryzen Threadripper PRO 7985WX, Linux 6.8.0-136-generic,
Rust 1.97.1 / LLVM 22.1.6, standard bench profile, no native-CPU or explicit
SIMD flags. The host was not isolated, core-pinned, or frequency-locked. See
the [performance and stat-arb audit](performance.md) for methodology,
confidence limits, Imperial HFT comparisons, semantic caveats, and the
visibility-versus-durability distinction.

## Record/replay smoke run

Commands:

```bash
cargo run --quiet -p hypercube-circuit --bin hypercube-replay -- \
  record-demo /tmp/hypercube-factor.jsonl 40 32

cargo run --quiet -p hypercube-circuit --bin hypercube-replay -- \
  verify /tmp/hypercube-factor.jsonl
```

The recording contained one manifest and 40 complete generations. Replay
recalculated 6,400 factor cells through a fresh engine and reproduced 1,280
trigger-state cells and all 78 transitions, with zero divergent generations.

## Semi-live smoke run

Command:

```bash
cargo run -p hypercube-engine --example synthetic_server -- \
  --address 127.0.0.1:18080 \
  --entities 8 \
  --interval-ms 75 \
  --slice-dir /tmp/hypercube-demo
```

The HTTP dashboard, `/api/snapshot`, and `/api/stream` all responded. One
observed generation contained:

```json
{
  "entity_count": 8,
  "nodes": 5,
  "values": 40
}
```

The publisher produced:

```text
catalog.json
layout.json
slices/dollar_volume_z.slice
slices/liquid_residual_score.slice
slices/price.slice
slices/residual_z.slice
slices/return.slice
```

Each `f64` slice was 320 bytes: a 256-byte versioned header plus eight
eight-byte values. The catalog exposed the same five node names and the layout
reported eight stable entities.

The live server uses mapped-memory visibility and does not synchronously flush
every generation. Its epoch makes each individual slice visible coherently;
recovery truth and multi-slice atomicity remain separate responsibilities.

## Financial terminal examples

Commands:

```bash
cargo run -q -p hypercube-engine --example etf -- \
  --record --ticks 1 --entities 160 --funds 12 \
  --top 5 --interval-ms 0 --seed 335342

cargo run -q -p hypercube-engine --example pairs -- \
  --record --ticks 1 --pairs 24 \
  --top 10 --interval-ms 0 --seed 335341
```

The ETF frame used 160 constituents to value 12 baskets. Its five-node graph
emitted 60 ETF-aligned cells and ranked premiums to basket value. The pairs
frame used 24 pair entities; its five-node graph emitted 120 cells and ranked
the absolute cointegrating residuals standardized against the preceding 20
observations.

The two committed [terminal recordings](markup/README.md) extend their recorded
seeds to 28 generations. They are functional examples, not evidence of
executable arbitrage after costs.

## Leakage-controlled stat-arb ML study

Commands:

```bash
python3 -m unittest discover -s experiments/statarb_ml -p 'test_*.py' -v
python3 experiments/statarb_ml/run.py
python3 experiments/statarb_ml/robustness.py --count 5
```

All seven causal/protocol tests passed. The detailed reference study used 20
pairs, 3,300 observations, two expanding folds, and distinct fit, validation,
calibration, selection, and untouched-test blocks separated by the full
12-observation label horizon. The robustness run repeated the complete study
over five deterministically derived seeds.

In the homogeneous stationary OU control, rolling-z had mean trade-return
t-statistic 12.666; the closest ML lane, LightGBM quantile, measured 12.648 and
produced 91.283 fewer normalized net-return units on average. In the synthetic
observable regime/heavy-tail positive control, CatBoost Huber measured 7.290
against rolling-z at 6.626, won all five seed comparisons, and added 48.176
normalized net-return units on average. LightGBM Huber also won all five, with
a +0.555 t-statistic delta and +32.926 mean total-net delta.

Quantile coverage and profit-class probability calibration transferred well
to the untouched reference tests, but calibration alone did not guarantee a
better selection policy. Results, exact seeds, package versions, fold
boundaries, plots, and caveats are committed under
[`experiments/statarb_ml/results`](../experiments/statarb_ml/results/). These
are normalized synthetic event results, not annualized Sharpe, formal
significance, or executable-arbitrage evidence.

## Packaging

`hypercube-slice` packages and verifies independently. `hypercube-engine`
contains its dashboard asset in the crate archive and is released after the
matching `hypercube-slice` version, because Cargo resolves published
dependencies during package verification. `hypercube-circuit` packages after
the matching engine version for the same reason.
