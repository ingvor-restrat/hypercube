# Slice + Hypercube

[Slice + Hypercube](hypercube_foundations.pdf)
([source](hypercube_foundations.tex)) is the public-repository edition of the
original strips/hypercube paper. It uses the current names `Slice`,
`hypercube-slice`, `hypercube-engine`, and the actual public Rust API.
Implemented behavior is separated from future generation manifests, function
cells, persistence, transport, and execution adapters.

The local class uses the Palatino/Helvetica typefaces from
`BRCentralRiskBook.tex` while retaining the StrategyNet cover, hierarchy,
spacing, and publication identity. The motivation section’s “Core thesis”
panel uses a light-gray treatment.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  hypercube_foundations.tex
```

`latexmk -c` removes intermediate files without deleting the PDF or source.
