# Reproducible Results

These functional results were recorded for Hypercube 0.1.0 on 24 July 2026.
They validate behavior and packaging; they are not performance benchmarks.

## Test suite

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Result:

| Suite | Tests | Result |
| --- | ---: | --- |
| engine, publisher, and synthetic injectors | 6 | passed |
| ETF and pairs example invariants | 2 | passed |
| slice layout, catalog, mmap, records, and vector algebra | 7 | passed |
| **Total** | **15** | **passed** |

The suite includes dependency-cycle rejection, stale-generation rejection,
tie-aware ranking, deterministic generic and OU market injection, long-only
ETF basket weights, reconstruction of the declared pair residual, aligned
publication, layout
mismatch rejection, live heartbeat observation, quote/trade/TAQ records, and
guarded dot products.

## Documentation contract

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Both crates build their complete public API documentation without warnings.
Missing public documentation is enabled at the crate level, and CI promotes
warnings to errors for the documentation build.

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
the absolute standardized cointegrating residuals.

The two committed [terminal recordings](markup/README.md) extend their recorded
seeds to 28 generations. They are functional examples, not evidence of
executable arbitrage after costs.

## Packaging

`hypercube-slice` packages and verifies independently. `hypercube-engine`
contains its dashboard asset in the crate archive and is released after the
matching `hypercube-slice` version, because Cargo resolves published
dependencies during package verification.
