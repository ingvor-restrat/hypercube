# Hypercube papers

## Foundations

[Hypercube: Foundations](hypercube_foundations.pdf)
([source](hypercube_foundations.tex)) is the public-repository edition of the
original strips/hypercube paper. It uses the current names `Slice`,
`hypercube-slice`, `hypercube-engine`, and the actual public Rust API. The
August 2026 revision adds reusable graph planning, online rolling moments,
publication durability policies, the Imperial HFT pattern audit, measured
Criterion results, and the leakage-controlled point/Huber/quantile/
triple-barrier stat-arb study.

## Callbacks, Triggers, and Replay

[Hypercube: Callbacks, Triggers, and Replay](hypercube_circuits.pdf)
([source](hypercube_circuits.tex)) shows how to run callbacks after each
completed Hypercube generation, maintain persistent trigger state, publish
state changes, and record and replay the calculation. It includes the ETF,
pairs, browser, and replay examples.

Both papers distinguish implemented behavior from future generation assembly,
general stateful function cells, transport adapters, and execution authority.

Both papers follow the plain 11-point article format used by the current Volt
papers: Palatino and Helvetica typefaces, compact margins, and a standard title
and abstract without a separate cover or contents page.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_foundations.tex

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_circuits.tex
```

`latexmk -c` removes intermediate files without deleting the PDF or source.
