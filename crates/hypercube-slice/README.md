# hypercube-slice

`hypercube-slice` provides typed, file-backed memory-mapped vectors with stable
entity layouts. It supports one writer and many readers, point-safe fixed
records, coherent `f64` vector snapshots, layout compatibility checks, and
ordinary vector operations.

This is the physical live-state layer used by the `hypercube` engine. See the
[repository README](https://github.com/ingvor-restrat/hypercube#readme) for
examples and architecture.
