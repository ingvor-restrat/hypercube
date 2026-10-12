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

The October 2026 revision adds the implemented indexed-projection runtime:
ordered picks, stateless calculations, independent output layouts, runtime graph
revisions, immutable shared results, and config recovery. It distinguishes these
in-process strips from memory-mapped slice files and general stateful cells.

## Callbacks, Triggers, and Replay

[Hypercube: Callbacks, Triggers, and Replay](hypercube_circuits.pdf)
([source](hypercube_circuits.tex)) shows how to run callbacks after each
completed Hypercube generation, maintain persistent trigger state, publish
state changes, and record and replay the calculation. It includes the ETF,
pairs, browser, and replay examples.

The October 2026 revision explains how projections consume coherent snapshots
and why their configs, activation history, and outputs are outside the current
Circuit recording contract. Its workspace test table includes the eight new
projection tests, for a total of 45.

Both papers distinguish implemented behavior from future generation assembly,
general stateful function cells, transport adapters, and execution authority.

Both papers use the repository's `strategynet-journal` class: compact A4 layout,
8-point body text, explicit two-column sections, a standard title and abstract,
and a two-column contents page.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_foundations.tex

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_circuits.tex
```

`latexmk -c` removes intermediate files without deleting the PDF or source.
