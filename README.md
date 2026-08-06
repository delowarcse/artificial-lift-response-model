# ASPM Surrogate Modeling - Artificial-Lift Setpoint Optimization

> Predicting the 48-hour liquid-rate response to a gas-lift setpoint change, and scoring
> candidate actions under **calibrated uncertainty** so only confident, in-support gains are recommended.

A surrogate model for a simplified Artificial-lift Setpoint Performance Modeling (ASPM) problem.
Each historical event is a real intervention on a gas-lifted well — the state before the change,
the action taken, and the observed 48-hour change in liquid production. The task is to learn the
response function **f(well_state, action) → Δ liquid rate**, then use it to score new candidate
actions safely.

---

## TL;DR

- **The response is driven by one lever** — the gas-injection change (**+0.39 m³/d per e3m³/d**,
  r = 0.635), stable across all four lift modes. The other two levers carry no measurable signal.
- **A deliberately simple model wins.** A quality-weighted **Ridge on 4 features** beats gradient
  boosting, random forest, and an all-features Ridge under group cross-validation.
- **Every prediction ships with a calibrated interval** via normalized split-conformal prediction
  (80% nominal → 0.803 empirical). Intervals widen on noisy wells, so the width itself is a trust signal.
- **The recommendation rule is uncertainty-first:** recommend only when the *entire* 80% interval is
  above zero. On 132 candidates it recommends **12** and holds **120**.

---

## Headline results

| Metric | Value |
|---|---|
| Dataset | 320 events · 12 wells · 4 lift modes · 132 candidates |
| Target | Δ liquid 48h (m³/d) · mean −0.59 · std 4.70 · skew −0.12 |
| Gas effect (the lever) | +0.39 m³/d per e3m³/d · r = 0.635 · stable 0.32–0.48 across modes |
| Final model gas coefficient | **+0.388** (recovers the physical slope) |
| **Group OOF (new-well)** | **R² 0.418 · MAE 2.52 · RMSE 3.58** |
| Random OOF (seen-well) | R² 0.406 · MAE 2.53 → **gap ≈ 0** |
| Noise ceiling | ~3.2 m³/d spread at near-zero action → achievable R² ≈ 0.5 (upper bound) |
| Conformal coverage | 80% → 0.803 · 90% → 0.903 |
| Interval width (adaptive) | contaminated 11.7 vs clean 8.0 m³/d |
| Candidate decisions | **12 recommend · 120 hold** |

The **near-zero gap between random and group CV** is the key result: the model transfers to wells
with no history because the gas effect is shared physics, not memorized per-well offsets.

---

## Repository structure

```
.
├── ASPM_Surrogate_Modeling_Challenge.ipynb   # main solution (fully commented, runs top-to-bottom)
├── simplified_aspm_events.csv                # 320 labeled training events
├── aspm_candidate_actions.csv                # 132 candidate actions to score (no target)
├── scored_candidate_actions.csv              # OUTPUT: candidates + predictions, intervals, decisions
└── README.md
```

---

## Approach

The notebook follows a standard, defensible DS workflow. Each step justifies the next.

### 1. EDA
Established the four facts that drive every downstream choice:
- **Gas action is the only lever with signal**; the slope is stable across lift modes (0.32–0.48),
  i.e. an approximately additive, low-interaction physical effect.
- **Afterflow / close-time changes carry no signal** (|r| ≤ 0.09) — a deliberate over-fitting trap.
- **The label is noisy** (~3.2 m³/d irreducible spread even at zero action) — a low-signal problem by design.
- **Confounders**: `downtime_flag` and `facility_constraint_flag` shift the response *and* co-occur
  with larger gas actions. All 132 candidates sit in the clean regime.

### 2. Validation strategy
Out-of-fold cross-validation with **three lenses**, because 320 rows over 12 wells is too small for a
single trustworthy holdout:

| Split | Tests | Deployment relevance |
|---|---|---|
| Random 5-fold | Interpolation, same wells in train & test | Re-optimizing wells we already operate (matches candidates) |
| **Group 5-fold by `well_id`** | Generalization to wells with no history | The conservative bound; gates recommendation safety |
| Time split (75/25) | Temporal drift | Whether a model fit on the past holds up later |

The same OOF residuals are reused to calibrate the conformal intervals. CV measures *predictive*
accuracy in the historical operating distribution — **not** the causal effect of an action (the label
is observational and confounded). That gap is why recommendations are uncertainty- and support-aware.

### 3. Model
A **quality-weighted Ridge** on four inputs — `action_delta_gas_e3m3d`, `current_lift_mode`,
`downtime_flag`, `facility_constraint_flag`. Chosen by evidence: richer alternatives
(all-24-features Ridge, HistGradientBoosting, RandomForest, extra action/context features) were
tested and **none beat it under group CV**. Rows are weighted by `data_quality_score` to down-weight
noisy events. The flags are included to *de-bias* the gas coefficient, never as levers.

### 4. Uncertainty
**Normalized split-conformal prediction intervals.** A shallow difficulty model σ(x) predicts the
error size from decision-time features; conformal calibration on `|residual| / σ` converts it into
intervals with a distribution-free marginal coverage guarantee. Normalizing by σ lets intervals widen
in noisy regimes instead of using one fixed width.

### 5. Diagnostics
Predicted-vs-actual, residual distribution, error by well / lift mode / quality / action size, and
uncertainty calibration — all on **group OOF** predictions (the conservative new-well view).

### 6. Candidate scoring & recommendation rule
Recommend an action only when **every** gate passes:
1. the entire 80% interval is above zero (`LCB80 > 0`) — a confident gain, not a hopeful mean,
2. the well state + action are **in historical support** (5-NN distance within training p95),
3. the gas change is within the historical p95 (16.1 e3m³/d) — no extrapolation,
4. the well is in the **clean operating regime** (flags are confounds, not levers),
5. context **data quality ≥ 0.60**.

Ranking is by the **lower confidence bound, never the raw predicted mean**.

---

## How to run

The notebook is self-contained. Place the two input CSVs alongside it and run top to bottom.

```bash
pip install numpy pandas scikit-learn scipy matplotlib
jupyter notebook ASPM_Surrogate_Modeling_Challenge.ipynb   # then: Restart & Run All
```

**Output:** running the final cell writes `scored_candidate_actions.csv` — the 132 candidates with
predicted response, 80% interval (`lcb80` / `ucb80`), support flags, and a `recommendation` +
human-readable `reason` for each.

> **Note on reproducibility:** exact uncertainty figures (R², interval widths, LB80 ordering) can
> shift by ~0.01 across scikit-learn versions, because `GroupKFold` fold composition and the
> gradient-boosted σ-model are version-sensitive. The point predictions and the overall narrative are
> stable. Numbers reported here reflect scikit-learn 1.x.

---

## Key output columns (`scored_candidate_actions.csv`)

| Column | Meaning |
|---|---|
| `pred_delta_liquid_m3d` | Expected 48h liquid gain |
| `lcb80` / `ucb80` | 80% lower / upper confidence bound (LCB drives the decision) |
| `interval_width` | Interval width — narrow = confident |
| `support_dist` / `in_support` | Distance to nearest training data / in-support flag |
| `large_gas_change` | Extrapolation guard (gas beyond training p95) |
| `recommendation` | `recommend` / `hold` |
| `reason` | Plain-English audit trail for the decision |

---

## Limitations

- **Observational label** → the model predicts associations, not guaranteed causal effects.
- **Two levers unmodeled** (afterflow, close-time) — correct on this data, a real gap if they matter physically.
- **Linear** — cannot capture saturation of the gas response at extreme deltas.
- **Conditional coverage** is imperfect (noisy tiers slightly under-covered); Mondrian conformal by
  tier would help with more data.
- **Only 12 wells** — encouraging but thin for fleet-wide generalization.

**Before field use:** interventional / staged-rollout data, more wells, coverage of the other levers,
a monitoring + recalibration loop, engineering-owned guardrails, and a human in the loop. This is one
input to a recommendation, not an autonomous controller.

---
