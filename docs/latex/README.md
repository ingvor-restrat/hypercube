# Slice + Hypercube papers

## Part I — Foundations

[Slice + Hypercube](hypercube_foundations.pdf)
([source](hypercube_foundations.tex)) is the public-repository edition of the
original strips/hypercube paper. It uses the current names `Slice`,
`hypercube-slice`, `hypercube-engine`, and the actual public Rust API.

## Part II — Circuits and Replay

[Slice + Hypercube II: Circuits and Replay](hypercube_circuits.pdf)
([source](hypercube_circuits.tex)) develops the stateful circuit layered after
Hypercube computation. It covers coherent snapshots, Disruptor callbacks,
backpressure, persistent triggers, exact record/replay, Aeron boundaries, and
the repository's ETF and pairs demonstrations.

Both papers distinguish implemented behavior from future generation assembly,
general stateful function cells, transport adapters, and execution authority.

The local class uses the Palatino/Helvetica typefaces from
`BRCentralRiskBook.tex` while retaining the StrategyNet cover, hierarchy,
spacing, and publication identity. The motivation section’s “Core thesis”
panel uses a light-gray treatment.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_foundations.tex

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_circuits.tex
```

`latexmk -c` removes intermediate files without deleting the PDF or source.
