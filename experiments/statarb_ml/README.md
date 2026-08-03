# Leakage-controlled stat-arb ML study

This experiment asks a narrow question: once a pairs strategy has declared a
causal mean-reversion trade, can tabular ML improve the decision to accept or
reject that event?

The answer is conditional. In the homogeneous stationary control, the simple
rolling-z policy remains the best average policy. In a synthetic positive
control with observable volatility regimes, persistence shifts, and
heavy-tailed shocks, robust point regression improves the rolling-z baseline
across all five deterministic seeds. That establishes that the harness can
detect useful conditional information; it does **not** establish that the same
information or profitability exists in market data.

The experiment is Python-only research tooling. It adds no Python or ML
dependency to Hypercube's Rust live path.

## Result first

The table reports five complete, independently generated two-fold studies.
The t-statistic is calculated over normalized net trade returns and is only a
descriptive comparison metric; it is not an annualized Sharpe ratio and does
not correct for cross-pair dependence.

| Scenario | Policy | Mean trade t-stat | Delta vs rolling-z | Seed wins | Mean total-net delta |
| --- | --- | ---: | ---: | ---: | ---: |
| stationary OU | rolling-z | 12.666 | — | — | — |
| stationary OU | closest ML lane, LightGBM quantile | 12.648 | -0.018 | 2/5 | -91.283 |
| regime switching | rolling-z | 6.626 | — | — | — |
| regime switching | CatBoost Huber | **7.290** | **+0.664** | **5/5** | **+48.176** |
| regime switching | LightGBM Huber | 7.181 | +0.555 | 5/5 | +32.926 |

The strongest conclusion is not that CatBoost always wins. It is that robust
regression helps when the data-generating process contains observable
conditional structure, while extra model capacity does not create incremental
value in the stationary control. Quantile and three-class models remain useful
for uncertainty and probability reporting, but neither was the best selection
policy here.

![Five-seed robustness result](results/statarb_ml_robustness.png)

The detailed reference-seed paths, fold metrics, calibration diagnostics, and
machine-readable manifests are in [`results/`](results/).

## Is ML used in the current pairs example?

No. The Rust `pairs` monitor uses a declared AR(1) residual process and a
fixed-capacity rolling estimator. It scores the current residual against the
previous 20 residuals before inserting the current observation. That is a
causal statistical monitor, not a fitted trading model.

ML enters this study one layer later. The economic hypothesis and side remain
fixed: a sufficiently large residual is traded toward its prior mean. A model
only ranks or rejects candidate events using information available at the
decision time. Keeping those responsibilities separate makes it possible to
measure whether ML adds anything beyond the z-score.

## What “calibration” means here

Several different operations are often called calibration. They should not
share data or be treated as interchangeable.

| Layer | Purpose | This study |
| --- | --- | --- |
| Residual calibration | Estimate hedge ratio, center, scale, and persistence. | The synthetic residual law is declared; every z-score uses only the preceding 20 observations. A real study must estimate hedge parameters inside each training fold. |
| Model fit | Learn return, quantile, or class score. | Expanding historical block. |
| Early stopping | Choose boosting iteration without fitting on later data. | A later validation block. |
| Predictive calibration | Correct point bias, quantile coverage, or class probability. | A still-later calibration block. |
| Policy calibration | Choose the minimum score at which a trade is accepted. | A separate selection block with a minimum trade count. |
| Evaluation | Measure the frozen pipeline. | An untouched test block. |

Every adjacent block is separated by a full 12-observation label horizon. The
five-stage order is therefore

```text
fit -> gap -> validation -> gap -> calibration -> gap -> selection -> gap -> test
```

No position may overlap another position in the same pair. Positions in
different pairs may overlap, which is one reason the reported trade t-statistic
must not be interpreted as a formal portfolio significance test.

Policy selection evaluates the fixed score quantiles 0%, 20%, 40%, 60%, 75%,
85%, and 90%. It maximizes selection-block trade-return t-statistic, then total
net return and trade count as deterministic tie-breaks, subject to at least 30
trades. Model scores must be nonnegative; rolling-z must also remain above the
original candidate threshold of 1.0. This small declared grid constrains, but
does not eliminate, threshold-selection variance.

## Causal residual and candidate event

For pair \(j\), let the residual be

\[
e_{j,t}=\log A_{j,t}-\alpha_j-\beta_j\log B_{j,t}.
\]

With preceding window \(W_{j,t}=\{e_{j,t-20},\ldots,e_{j,t-1}\}\), the
candidate score is

\[
z_{j,t}=\frac{e_{j,t}-\mu(W_{j,t})}{\sigma(W_{j,t})}.
\]

The current residual is not part of its own center or scale. An event becomes
a candidate when \(|z_{j,t}|\ge 1\), and its fixed side is
\(s_{j,t}=-\operatorname{sign}(z_{j,t})\). Features contain the current
causal z-score, one/three/ten-step residual changes, a 5-to-20-step volatility
ratio, a trailing AR(1) estimate and half-life, and an observable market
volatility proxy. The latent synthetic regime and arbitrary pair identity are
not features.

## Triple barrier as a three-way target

Yes: triple barrier naturally creates a three-class first-touch problem. For
future offset \(h\), define normalized gross trade return

\[
g_{j,t}(h)=s_{j,t}\frac{e_{j,t+h}-e_{j,t}}{\sigma(W_{j,t})}.
\]

With profit barrier \(b_+=0.75\), stop barrier \(b_-=1.00\), and vertical
horizon \(H=12\), define

\[
\tau_+=\inf\{h\in[1,H]:g(h)\ge b_+\},\qquad
\tau_-=\inf\{h\in[1,H]:g(h)\le-b_-\}.
\]

The first observed touch wins:

\[
y_{j,t}=\begin{cases}
+1,&\tau_+<\tau_-\text{ and }\tau_+\le H,\\
-1,&\tau_-<\tau_+\text{ and }\tau_-\le H,\\
0,&\text{neither barrier is touched before }H.
\end{cases}
\]

The regression target uses the same exit but retains magnitude:

\[
r_{j,t}=g_{j,t}(\min(\tau_+,\tau_-,H))-c,
\qquad c=0.08.
\]

This distinction matters. A classifier estimates which boundary is reached;
a regressor estimates economic return. Classification discards within-class
overshoot and the variable vertical-barrier payoff, so the calibrated class
probabilities are converted back to expected net return using class-conditional
payoffs estimated only on the calibration block.

## Models and their calibration

### Point regression

LightGBM L2 and CatBoost RMSE estimate a conditional mean-like return. Their
calibration-block mean residual supplies one frozen intercept correction:

\[
\delta=\operatorname{mean}(r-\hat r),\qquad
\hat r^{\mathrm{cal}}(x)=\hat r(x)+\delta.
\]

The Huber lanes replace squared loss with a quadratic center and linear tails.
They are the most directly useful choice here when bad regimes contain
heavy-tailed shocks: both Huber implementations beat rolling-z in all five
positive-control seeds.

### Quantile regression

LightGBM fits separate 20th, 50th, and 80th percentile models. CatBoost fits
the same three outputs jointly with `MultiQuantile`. For each quantile \(q\),
a split marginal correction is learned on the calibration block:

\[
\Delta_q=\operatorname{Quantile}_q(r-\widehat Q_q(x)),\qquad
\widetilde Q_q(x)=\widehat Q_q(x)+\Delta_q.
\]

Corrected outputs are sorted per row to prevent quantile crossing. The frozen
selection score is a pre-declared lower-tail utility,

\[
u_Q=\widetilde Q_{0.50}
 -0.25\bigl(\widetilde Q_{0.50}-\widetilde Q_{0.20}\bigr).
\]

A strict rule requiring \(\widetilde Q_{0.20}>0\) is not useful for this
barrier geometry: in the reference tests, it accepts essentially no events.
The calibrated test coverages remain close to their targets: roughly
0.18–0.20, 0.47–0.49, and 0.79–0.80 for the three quantiles.

### Three-class barrier classifier

LightGBM and CatBoost fit stop/vertical/profit classes with training-only
class weights. Because weighted classifiers should not be trusted as
probability estimators directly, an independent calibration block fits
one-vs-rest sigmoid calibrators. The economic score is

\[
u_C(x)=\sum_{k\in\{-1,0,+1\}}
P^{\mathrm{cal}}(y=k\mid x)\,\bar r_k,
\]

where \(\bar r_k\) is the calibration-block mean net payoff for class \(k\).
Reference-seed expected calibration error for the profit class is about
0.03 on untouched tests. Good probability calibration did not, by itself,
make the classifier a better trade selector.

## LightGBM or CatBoost?

Both are current, capable choices; the target and validation design matter
more than the logo.

- LightGBM offers L2, Huber, quantile, and multiclass objectives and is a
  straightforward fast baseline. Separate quantile models can be trained or
  served independently.
- CatBoost offers the same broad families plus a joint `MultiQuantile` loss
  and strong native categorical handling. Its joint multi-quantile objective
  is CPU-only in the current official support table.
- CatBoost's `has_time=True` is used here so it preserves input order rather
  than randomly permuting rows. This is useful, but it does not replace
  chronological splits or horizon purging.
- For this controlled stress, CatBoost Huber has the strongest replicated
  selection result. That is evidence for this data-generating process, not a
  universal library ranking.

Relevant upstream references are the
[LightGBM objective documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
[CatBoost regression losses](https://catboost.ai/docs/en/concepts/loss-functions-regression),
[CatBoost time-order parameter](https://catboost.ai/docs/en/references/training-parameters/common),
and scikit-learn's
[probability-calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
The triple-barrier formulation follows Marcos López de Prado's *Advances in
Financial Machine Learning*; it is a label construction, not protection
against leakage by itself.

## Where probabilistic Torch and graph models fit

The protocol is model-agnostic. Volt's DeepSets or GraphNN can consume the same
causal event contract when relationships across sets, paths, or graph
neighborhoods carry information that a row-wise tree cannot represent. Keep
the rolling-z lane and boosted-tree lanes as mandatory controls; a neural model
should earn its additional data, serving, and checkpoint complexity.

Probabilistic Torch is technically straightforward:

- three logits plus softmax model the stop/vertical/profit target;
- three ordered heads with pinball loss model conditional quantiles;
- a location and positive scale can parameterize a Gaussian or Student-t
  return likelihood; and
- ensembles or dropout can supply epistemic diagnostics, though neither is
  automatically a calibrated probability.

The same independent calibration and selection blocks still apply. A neural
checkpoint should include feature schema, normalization state, target/barrier
definition, calibration map, selection threshold, library versions, and a
content digest. In a production split, Rust can continue to own causal rolling
features and the event pipeline while Python/Torch owns training; inference can
move behind a versioned native boundary only after profiling shows that Python
serving is material.

## Scenario design

The study deliberately includes one negative and one positive control.

- `stationary_ou` uses a homogeneous Gaussian mean-reverting residual law.
  Extra features contain little incremental structure; ML should not win
  merely by recreating the z-score.
- `regime_switching` uses three persistence/volatility states. Stressed states
  have unit-variance Student-t(4) idiosyncratic shocks, and the breakdown state
  is mildly explosive. A noisy, causal volatility proxy is visible; the true
  regime is retained only for diagnostics. This is constructed so a conditional
  filter *can* help.

The positive-control transition matrix is

\[
P=\begin{bmatrix}
0.965&0.030&0.005\\
0.060&0.890&0.050\\
0.150&0.100&0.750
\end{bmatrix}.
\]

State zero uses each pair's base persistence and volatility. State one adds
0.10 to persistence (capped at 0.985), multiplies innovation scale by 1.8,
and uses standardized Student-t(4) idiosyncratic noise. State two sets
persistence to 1.025, multiplies scale by 2.5, and retains the same heavy-tailed
noise. A smoothed same-time common-volatility measurement is available to the
models; the state itself is not. The stationary control instead fixes
\(\phi=0.90\), \(\sigma=0.004\), and Gaussian noise for every pair.

All returns are normalized residual units. Costs, borrow constraints, market
impact, parameter-estimation error, asynchronous legs, portfolio capital, and
real market microstructure are not modeled. These are controlled algorithm
tests, not executable-arbitrage results.

## Reproduce it

Install the pinned research dependencies in an isolated environment, then run:

```bash
python3 -m unittest discover -s experiments/statarb_ml -p 'test_*.py' -v
python3 experiments/statarb_ml/run.py
python3 experiments/statarb_ml/robustness.py --count 5
```

`run.py` writes one detailed two-fold reference study. `robustness.py` derives
its additional seeds deterministically with NumPy `SeedSequence.spawn` and
writes the cross-seed comparison. Use `--quick` for a smoke-sized run.

Committed artifacts:

- [`summary.csv`](results/summary.csv) and
  [`fold_metrics.csv`](results/fold_metrics.csv): frozen reference-test trade
  outcomes;
- [`calibration_diagnostics.csv`](results/calibration_diagnostics.csv): point,
  quantile, and probability diagnostics;
- [`robustness_runs.csv`](results/robustness_runs.csv) and
  [`robustness_summary.csv`](results/robustness_summary.csv): five-seed evidence;
- [`manifest.json`](results/manifest.json) and
  [`robustness_manifest.json`](results/robustness_manifest.json): exact protocol,
  seeds, feature names, and package versions;
- [`statarb_ml_results.png`](results/statarb_ml_results.png),
  [`statarb_ml_calibration.png`](results/statarb_ml_calibration.png), and
  [`statarb_ml_robustness.png`](results/statarb_ml_robustness.png): paper-ready
  plots.

## Recommended next experiment

Do not replace the Rust z-score monitor with a Python model on the strength of
this synthetic result. Freeze the same feature and label contract, estimate
alpha/beta inside each fold, and run it over a survivorship-controlled market
dataset with bid/ask execution, borrow, and pair-level clustered or block
bootstrap uncertainty. Promote a model only if its improvement survives time,
universe, cost, and calibration drift. The likely production shape is a small
versioned model artifact behind the existing Rust candidate pipeline, with a
rolling-z fallback and explicit abstention when features or calibration are
stale.
