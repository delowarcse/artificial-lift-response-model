# ASPM Surrogate Modeling — Code Walkthrough

A cell-by-cell, line-by-line guide to `ASPM_Surrogate_Modeling_Challenge.ipynb`, Each section has:

- **What the code does** (line by line where it's non-obvious)
- **Why it's there** (the reasoning an interviewer wants to hear)
- **Likely questions** they'll ask, with crisp answers

---

## 0. The 60-second story

> "Each row is a historical well intervention: the pre-change state `s`, the setpoint action `x`,
> and the observed 48-hour change in liquid rate. I built a surrogate `f(s, x) → Δliquid`.
> The EDA showed the response is basically a single **linear lever** — the gas-injection change,
> ~**+0.39 m³/d per e3m3d** — so I built a **parsimonious quality-weighted Ridge** on 4 features and
> proved that boosting/RF/all-features don't beat it under **group cross-validation by well**.
> I added **normalized split-conformal intervals** for honest, adaptive uncertainty, and a
> **recommendation rule driven by the lower confidence bound**, not the mean — so I only recommend
> an action when the whole 80% interval sits above zero, the point is in-support, the gas change
> isn't an extrapolation, the regime is clean, and the data quality is acceptable. Out of 132
> candidates it recommends 12."

Headline numbers to memorize:
- **320 events, 12 wells, 4 lift modes**, time span 2023-12 → 2026-04.
- Gas effect: **slope ≈ +0.39 m³/d per e3m3d**, Pearson **r ≈ 0.63**.
- Group OOF: **MAE 2.52, RMSE 3.58, R² 0.418**; random OOF nearly identical (**MAE 2.53, R² 0.406**).
- Noise floor ≈ **3.2 m³/d** even at near-zero action → **R² ceiling ≈ 0.5** (an upper bound on noise, so the true ceiling is ≥ that).
- Conformal coverage: 80% nominal → **0.803** empirical; 90% → **0.903**.
- Candidates: **132**, all in-support, only 6 of 12 wells, skew to positive gas. **12 recommended.**

---

## Cell 2 — Imports & global config

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', 100)
RANDOM_STATE = 42
```

- **`ColumnTransformer` / `Pipeline`**: the whole preprocessing + model is one object, so **fit happens
  inside each CV fold** — no leakage of scaler/imputer statistics from test into train.
- **`GroupShuffleSplit` / (later) `GroupKFold`**: the group machinery is imported up front because
  grouping by well is the backbone of the validation story.
- **`RANDOM_STATE = 42`**: reproducibility. Same folds, same numbers every run.
- **Talking point:** everything here signals "leakage-safe pipeline + group-aware validation" before a
  single model is trained.

---

## Cell 4 — Load data

```python
events_path = 'simplified_aspm_events.csv'
candidates_path = 'aspm_candidate_actions.csv'
df = pd.read_csv(events_path, parse_dates=['event_time'])
candidate_actions = pd.read_csv(candidates_path)
print(df.shape); display(df.head())
print(candidate_actions.shape); display(candidate_actions.head())
```

- **`parse_dates=['event_time']`**: needed so the later **time-ordered split** can sort chronologically.
- `df` = the 320 labeled training events; `candidate_actions` = 132 hypothetical actions **with no target**.
- **Q: Why keep them separate?** Candidates have no `target_*` column — they're what we score at the end.

---

## Cell 6 — EDA: numeric summaries

```python
target_col = 'target_delta_liquid_48h_m3d'
print(df[target_col].describe().round(3))
print(f"Skew={df[target_col].skew():.2f}  |  NaN in target: {df[target_col].isna().sum()}")
```
- Establishes the target is **roughly symmetric** (skew small), **mean ≈ -0.6, std ≈ 4.7**, **no missing targets**.
- Symmetric + no NaNs → **plain regression, no target transform** is justified.

```python
print(df['event_quality_label'].value_counts(dropna=False))
miss = df.isna().mean(); print(miss[miss > 0].sort_values(ascending=False).round(4))
cmiss = candidate_actions.isna().mean(); print(cmiss[cmiss > 0]...)
```
- **Quality distribution**: clean 228 / low_quality 60 / contaminated 32 → motivates **sample weighting**.
- **Missingness**: only a few context features, all **<4%** → simple imputation is fine; don't over-engineer.

```python
print('Candidate wells:', sorted(candidate_actions['well_id'].unique()),
      '| all seen in training:', set(candidate_actions['well_id']) <= set(df['well_id']))
```
- Confirms every candidate well **appears in training** — this is why the **random-CV** number is the
  relevant deployment analog for candidates (we're re-optimizing wells we already operate).

```python
print(df.groupby('event_quality_label')[target_col].agg(['count','mean','std']))
print(df.groupby('current_lift_mode')[target_col].agg(['count','mean','std']))
print(df.groupby('downtime_flag')[target_col].agg(['count','mean']))
print(df.groupby('facility_constraint_flag')[target_col].agg(['count','mean']))
```
- **Confounder detection.** `downtime_flag=1` events average **+2.3 m³/d** vs **−0.9** for normal events (gap ≈ 3.2); `facility_constraint_flag=1`
  shifts it down. These are the "intended traps" — associated with the response *and* with bigger gas
  actions, so they confound the action effect.

```python
for m, g in df.groupby('current_lift_mode'):
    b = np.polyfit(g['action_delta_gas_e3m3d'], g[target_col], 1)
    print(f'  {m}  n={len(g)}  slope={b[0]:+.3f}')
```
- **Key move:** fits the gas→target slope *within each lift mode*. `np.polyfit(..., 1)` = degree-1 OLS;
  `b[0]` is the slope. Result: slope is **stable ~0.32–0.48 across modes** → the gas effect is a
  **shared, near-additive physical relationship**, not a per-mode quirk. This is the entire justification
  for a simple global linear model.

```python
num = df.select_dtypes(include=[np.number])
print(num.corr()[target_col].drop(target_col).sort_values(key=np.abs, ascending=False))
```
- Ranks features by |correlation| with target. Gas action dominates; afterflow/close-time ≈ 0.
- **`key=np.abs`**: sort by magnitude so strong negative and positive correlations both float to the top.

**Likely questions**
- *"How did you know a linear model would work?"* → the within-mode slope stability + the r≈0.63 gas
  correlation + flat afterflow/close-time. It's evidence-led, not assumed.
- *"What are the confounders and why do they matter?"* → downtime/facility flags move both the response
  and the action size; I include them so the model can *explain them away* in training, but I never use
  them as levers for recommendations (all candidates are in the clean regime anyway).

---

## Cell 7 — EDA: the 9-panel figure

Setup:
```python
from scipy.spatial.distance import cdist
QCOL = {'clean':'#2a9d8f','low_quality':'#e9c46a','contaminated':'#e76f51'}
LM = sorted(df['current_lift_mode'].unique())
LMC = dict(zip(LM, [...colors...]))
fig, ax = plt.subplots(3, 3, figsize=(16, 13))
```
- `QCOL` / `LMC`: fixed color keys so quality and lift mode read consistently across all panels.

Panels (what each *proves*):
1. **Target distribution by quality** — symmetric, centered near 0; contaminated is visibly wider.
2. **Gas action vs target** with an overlaid `np.polyfit` line — the dominant driver, slope shown, **r≈0.63**.
3. **Afterflow / close-time deltas vs target** — flat cloud, **|r| ≤ 0.09** (afterflow −0.004, close-time −0.089), i.e. no meaningful signal (the overfit trap).
4. **Target boxplots by lift mode** — structural per-mode offsets (plunger runs low).
5. **Per-well mean ± std** (`barh` with `xerr`) — real per-well heterogeneity → motivates group CV.
6. **Confounders** — boxplots split by downtime/facility flags; downtime=1 averages +2.3 vs −0.9 (gap ≈ 3.2).
7. **Candidate vs training gas distribution** — candidates **skew positive**; important for the
   extrapolation guardrail later.
8. **Support via 5-NN distance** — the important one, see below.
9. **Correlation bar chart** — same ranking as Cell 6, visual.

The support computation (panel 8) — worth understanding line by line because the same idea reappears in the uncertainty and scoring cells:
```python
E = df[featnum + ae]; C = candidate_actions[featnum + ac].copy(); C.columns = featnum + ae
mu, sd = E.mean(), E.std().replace(0, 1)
Ez = ((E - mu)/sd).fillna(0).values; Cz = ((C - mu)/sd).fillna(0).values
dee = cdist(Ez, Ez); np.fill_diagonal(dee, np.inf)
nn_tr = np.sort(dee, 1)[:, :5].mean(1)
nn_cd = np.sort(cdist(Cz, Ez), 1)[:, :5].mean(1)
```
- **Standardize** training (`Ez`) and candidates (`Cz`) using **training** mean/std → distances are
  comparable across features with different units. `sd.replace(0,1)` avoids divide-by-zero on constant cols.
- **`cdist`** = pairwise Euclidean distances. `np.fill_diagonal(dee, np.inf)` removes each train row's
  self-match (distance 0) so a point isn't its own neighbor.
- **`np.sort(...,1)[:, :5].mean(1)`** = mean distance to the **5 nearest** training events = a "how novel
  is this point" score. Candidates compare against training rows (`cdist(Cz, Ez)`).
- Conclusion printed at the end: **0 candidates beyond training p99** → all candidates are in-support, so
  support distance *alone* can't separate good from bad candidates → the rule must lean on **uncertainty
  and action size** too.

**Likely questions**
- *"What's 'support' and why compute it?"* → nearest-neighbor distance in standardized feature space; a
  cheap out-of-distribution / extrapolation flag. If a candidate is far from all training data, the
  model is extrapolating and I shouldn't trust it.
- *"Why 5 neighbors?"* → smooths out single-point noise; robust to one coincidental near-duplicate. Not
  sensitive — 3 or 10 give the same qualitative answer.

---

## Cell 10 — Target / features / metadata split (the leakage guard)

```python
target_col = 'target_delta_liquid_48h_m3d'
metadata_cols = ['event_id', 'event_time', 'event_quality_label']
feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]
X = df[feature_cols].copy(); y = df[target_col].copy()
```
- **The critical decision:** `event_quality_label` is **post-hoc** — it's assigned *after* observing the
  event, so using it as a model input would be **leakage** (you don't know an event is "contaminated"
  until after it happens). It's held out as metadata and used **only for sample weighting and
  diagnostics**, never as a causal feature.
- `event_id` / `event_time` are identifiers, not features.

```python
groups = df['well_id'].values                     # group CV by well
sample_weight = df['data_quality_score'].values   # down-weight noisy events
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]
```
- `groups` feeds `GroupKFold`. `sample_weight` = `data_quality_score` (a *pre-event* quality proxy, so
  it's safe to use, unlike the post-hoc label).

```python
post_event = {'event_id','event_time','event_quality_label', target_col}
assert not (post_event & set(feature_cols)), 'post-event column leaked into features'
```
- An **executable leakage test.** If any post-event column ever slips into the feature list, the notebook
  fails loudly instead of silently training on leaked info. Great thing to point at in the interview.

**Likely questions**
- *"Why is quality label leakage but data_quality_score isn't?"* → `data_quality_score` is available at
  decision time (a sensor/telemetry quality proxy you know *before* acting); `event_quality_label` is a
  retrospective judgment of how the event turned out. Using the latter as a feature would let the model
  peek at the future.

---

## Cell 11 — Shared modeling helpers

```python
def build_preprocessor(X):
    cat = X.select_dtypes(include=['object','category']).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)])
```
- Numerics: **median impute** (robust to outliers) then **standardize**. Categoricals: **most-frequent
  impute** then **one-hot**.
- **`handle_unknown='ignore'`** is the key line: under group CV a fold's test well may be a category the
  fold never saw; instead of crashing, unseen levels encode as all-zeros and the prediction **degrades
  to the learned baseline** — exactly the new-well behavior we want to *measure*, not error out on.

```python
def regression_metrics(y_true, y_pred, label='model', verbose=True):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    ...
    return {'label':label,'MAE':mae,'RMSE':rmse,'R2':r2}
```
- One consistent metric bundle everywhere. **MAE** (robust, interpretable in m³/d), **RMSE** (penalizes
  big misses), **R²** (variance explained vs the mean baseline).

```python
def oof_predictions(estimator_factory, X, y, cv, groups=None, weight=None):
    oof = np.full(len(y), np.nan)
    splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for tr, te in splits:
        est = estimator_factory()
        if weight is not None:
            try:    est.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=weight[tr])
            except TypeError: est.fit(X.iloc[tr], y.iloc[tr])
        else:       est.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = est.predict(X.iloc[te])
    return oof
```
- **The workhorse.** Produces **out-of-fold predictions**: every row is predicted by a model that
  **did not train on it** (and under GroupKFold, never saw its well).
- **`estimator_factory`** is a *function* returning a fresh pipeline, so each fold gets a clean,
  un-leaked model.
- **`model__sample_weight=weight[tr]`**: the `model__` prefix routes the weight through the Pipeline to
  the final estimator step named `'model'`. The `try/except TypeError` gracefully skips weighting for
  estimators that don't accept it.
- Why OOF instead of a single split: with 320 rows every row contributes a residual → far lower-variance
  model selection, **and** those same OOF residuals are **reused to calibrate the conformal intervals**.

```python
kf  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # seen-well interpolation
gkf = GroupKFold(n_splits=5)                                      # generalization to NEW wells
```
- Two CV objects, two questions: `kf` = re-optimizing known wells; `gkf` = brand-new well.

**Likely questions**
- *"What does `model__sample_weight` do?"* → sklearn's double-underscore parameter routing; sends the
  weight to the pipeline's `model` step at fit time, inside each fold (so weighting never leaks).
- *"Why out-of-fold rather than `cross_val_score`?"* → I need the actual per-row predictions/residuals
  for the conformal calibration and the diagnostics, not just an averaged score.

---

## Cell 13 — Validation strategy: random vs group vs time

```python
ridge_all = lambda: Pipeline([('prep', build_preprocessor(X)), ('model', Ridge(alpha=5.0))])
dummy     = lambda: Pipeline([('prep', build_preprocessor(X)), ('model', DummyRegressor(strategy='mean'))])
```
- Two `lambda` factories (fresh pipeline per fold). `DummyRegressor(strategy='mean')` = the **no-skill
  floor** (always predicts the training mean, R²≈0 by construction).

```python
# RANDOM 5-fold
regression_metrics(y, oof_predictions(dummy,     X, y, kf), '  Mean baseline')
regression_metrics(y, oof_predictions(ridge_all, X, y, kf, weight=sample_weight), '  Ridge (all features)')
# GROUP 5-fold
regression_metrics(y, oof_predictions(dummy,     X, y, gkf, groups=groups), '  Mean baseline')
regression_metrics(y, oof_predictions(ridge_all, X, y, gkf, groups=groups, weight=sample_weight), '  Ridge (all features)')
```
- Runs the same models under both CV schemes. **The finding that matters:** random-R² ≈ group-R² (~0.34
  for Ridge). **Small gap = the signal is a universal action effect, not memorized per-well offsets** →
  the model transfers to new wells.

```python
order = np.argsort(df['event_time'].values); cut = int(len(order)*0.75)
tr_t, te_t = order[:cut], order[cut:]
rid = ridge_all(); rid.fit(X.iloc[tr_t], y.iloc[tr_t], model__sample_weight=sample_weight[tr_t])
regression_metrics(y.iloc[te_t], rid.predict(X.iloc[te_t]), '  Ridge (all features)')
```
- **Time split:** train on the earliest 75% by `event_time`, test on the latest 25%. A **drift check** —
  does a model fit on the past hold on later events? It does (≥ as good), so no material temporal drift.

```python
for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
    print(f'  fold {i}: {sorted(set(df.iloc[te].well_id))}')
```
- Prints which **wells** are held out in each group fold — proof the folds are genuinely well-disjoint.

**Likely questions**
- *"Why three splits?"* → they answer three different deployment questions: re-optimizing known wells
  (random), a new well (group), and the future (time). Group is the conservative bound I gate safety on.
- *"Why is `alpha=5.0` here but `alpha=1.0` in the surrogate?"* → this is the throwaway all-features
  baseline; more features + collinearity need more regularization. The final 4-feature surrogate is
  low-collinearity so less shrinkage is optimal (and performance is flat over alpha ∈ [0.3, 10]).

---

## Cell 15 — Baselines (does complexity earn its keep?)

```python
mean_base = lambda: Pipeline([('prep', build_preprocessor(X)), ('model', DummyRegressor(strategy='mean'))])
gas_only  = lambda: Pipeline([('prep', ColumnTransformer([('gas','passthrough',['action_delta_gas_e3m3d'])])),
                              ('model', LinearRegression())])
ridge_all = lambda: Pipeline([('prep', build_preprocessor(X)), ('model', Ridge(alpha=5.0))])
```
- Three baselines of increasing sophistication:
  1. **Mean** — no-skill floor.
  2. **Gas-only OLS** — `'passthrough'` sends *only* `action_delta_gas_e3m3d` to a plain
     `LinearRegression`. The physically-motivated "how far does the one lever get us?" baseline.
  3. **Ridge on all features** — the naive "throw everything in" baseline.

```python
for name, fac in [('Mean', mean_base), ('Gas-only OLS', gas_only), ('Ridge (all feats)', ridge_all)]:
    g = regression_metrics(y, oof_predictions(fac, X, y, gkf, groups=groups, weight=sample_weight), verbose=False)
    r = regression_metrics(y, oof_predictions(fac, X, y, kf,               weight=sample_weight), verbose=False)
    print(f'{name} {g["R2"]} {g["MAE"]} {r["R2"]} {r["MAE"]}')
```
- Prints group vs random R²/MAE side by side. **Punchline:** gas-only already gets **R²≈0.38** and
  performs the *same* on new wells as seen wells — because the gas effect is universal physics, not a
  per-well trick. Adding all features barely moves it. This sets up "the simple model wins."

**Likely questions**
- *"Why bother with a gas-only baseline?"* → it isolates the dominant lever and gives an honest bar. If
  my final model can't clearly beat one feature, the extra complexity isn't justified.

---

## Cell 17 — The surrogate model (the centerpiece)

```python
SURR_NUM = ['action_delta_gas_e3m3d', 'downtime_flag', 'facility_constraint_flag']
SURR_CAT = ['current_lift_mode']
def make_surrogate():
    prep = ColumnTransformer([('num', StandardScaler(), SURR_NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), SURR_CAT)])
    return Pipeline([('prep', prep), ('model', Ridge(alpha=1.0))])
```
- **The final model: 4 inputs.** The **gas action** (the real lever) + **lift mode** (structural
  per-mode offsets, one-hot) + the **two confounder flags** (so the model can explain away downtime /
  facility effects instead of attributing them to the action).
- **Deliberately excluded:** afterflow/close-time deltas (no signal) and the ~20 context features (tested,
  added variance not signal). `StandardScaler` → coefficients are comparable; `Ridge(alpha=1.0)` → mild
  shrinkage for the small gas↔downtime collinearity.

```python
histgbr = lambda: Pipeline([('prep', build_preprocessor(X)),
             ('model', HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                       min_samples_leaf=20, l2_regularization=1.0, random_state=RANDOM_STATE))])
rf = lambda: Pipeline([('prep', build_preprocessor(X)),
             ('model', RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE))])
```
- **Comparators** — gradient boosting and random forest, both regularized (shallow depth, min-leaf
  constraints). The point is to *show* they don't beat the linear model, not to tune them to death.

```python
for name, fac in [('Ridge (all feats)', ridge_all), ('HistGradientBoosting', histgbr),
                  ('RandomForest', rf), ('Surrogate: parsimonious linear', make_surrogate)]:
    g = ...gkf...; r = ...kf...
    print(...)
```
- Head-to-head under both CV schemes. **Result: the 4-feature linear model wins/ties on group CV**
  (R²≈0.41) — evidence-led model selection, the assignment's "simple model + strong validation" outcome.

```python
surrogate_oof        = oof_predictions(make_surrogate, X, y, gkf, groups=groups, weight=sample_weight)
surrogate_oof_random = oof_predictions(make_surrogate, X, y, kf,               weight=sample_weight)
surrogate_pred = surrogate_oof
surrogate = make_surrogate().fit(X, y, model__sample_weight=sample_weight)
```
- Stores **group OOF** (conservative / new-well) and **random OOF** (seen-well ≈ candidate regime) for
  reuse downstream (conformal + diagnostics). Then fits the **final surrogate on ALL data**
  (quality-weighted) — this is the model that scores candidates.

```python
scaler = surrogate.named_steps['prep'].named_transformers_['num']
rmodel = surrogate.named_steps['model']
ohe    = surrogate.named_steps['prep'].named_transformers_['cat']
names  = SURR_NUM + list(ohe.get_feature_names_out(SURR_CAT))
for n_, c_ in zip(names, rmodel.coef_): print(f'  {n_} {c_:+.3f}')
print(f'=> Gas effect on original scale: {rmodel.coef_[0] / scaler.scale_[0]:+.3f} m3/d per e3m3d ...')
```
- **Interpretability payoff.** Digs the fitted objects out of the pipeline and prints coefficients.
- **The clincher line:** `coef_[0] / scaler.scale_[0]` **un-standardizes** the gas coefficient back to
  physical units → **≈ +0.39 m³/d per e3m3d**, matching the raw EDA slope. The model's learned parameter
  equals the physics — that's the sentence to say out loud.

**Likely questions**
- *"Why not just use the boosted model — it's usually more accurate?"* → On group CV it isn't, here. The
  signal is low and near-linear; extra flexibility fits noise (afterflow/close-time) and adds variance.
  With 320 rows and 12 wells, parsimony generalizes better and is auditable for a safety-gated decision.
- *"Why include confounders you won't use as levers?"* → to *de-confound* the training fit — otherwise
  the gas coefficient absorbs the downtime rebound (+2.3 vs −0.9) and is biased. I include them to estimate the gas
  effect cleanly, then hold candidates to the clean regime so I never act on a confounder.
- *"Why divide by `scaler.scale_[0]`?"* → coefficients are on standardized inputs; dividing by the
  feature's std converts back to per-unit-of-gas physical units.

---

## Cell 19 — Uncertainty: normalized split-conformal intervals

The idea in one breath: **conformal prediction turns residuals into intervals with a distribution-free
coverage guarantee; "normalized" means the interval width scales with a per-point difficulty estimate
`σ(x)`, so it's wide where the model is genuinely unsure and tight where it's reliable.**

```python
oof_resid = y.values - surrogate_oof            # honest new-well residuals
```
- Calibrate on **group OOF** residuals so intervals reflect **new-well** error, not memorized wells.

Support distance (reused from EDA, now as a function):
```python
_supp_mu = df[SUPP_CTX+SUPP_ACT].mean(); _supp_sd = df[SUPP_CTX+SUPP_ACT].std().replace(0,1)
_supp_Z  = ((df[SUPP_CTX+SUPP_ACT]-_supp_mu)/_supp_sd).fillna(0).values
def support_distance(frame, is_train=False):
    Q = ((frame[...]-_supp_mu)/_supp_sd).fillna(0).values
    d = cdist(Q, _supp_Z)
    if is_train: np.fill_diagonal(d, np.inf)   # drop self-match for training rows
    return np.sort(d, axis=1)[:, :5].mean(1)
supp_train = support_distance(df, is_train=True)
SUPP_P95 = np.percentile(supp_train, 95)       # out-of-support threshold used at scoring time
```
- Same 5-NN standardized-distance idea; `is_train` toggles the self-match removal. `SUPP_P95` is the
  guardrail threshold for candidates.

The difficulty model `σ(x)`:
```python
def difficulty_features(frame, supp):
    return pd.DataFrame({
        'abs_gas_action': frame['action_delta_gas_e3m3d'].abs().values,
        'data_quality_score': ..., 'production_volatility': ..., 'liquid_loading_risk': ...,
        'support_distance': supp, 'downtime_flag': ..., 'facility_flag': ..., 'watercut': ...})
```
- **Every difficulty feature is known at decision time for a candidate** — that's deliberate, so `σ`
  can be evaluated on candidates that have no target. It predicts *how big the error will be*, not the
  response.

```python
SIGMA_FLOOR = 0.25 * np.median(np.abs(oof_resid))   # keep sigma away from 0 so ratios stay finite
def _fit_sigma(feats, resid):
    gb = HistGradientBoostingRegressor(max_depth=2, max_iter=150, learning_rate=0.05,
                                       min_samples_leaf=25, l2_regularization=1.0, random_state=RANDOM_STATE)
    return gb.fit(feats, np.abs(resid))          # regress |residual| on difficulty features
```
- The difficulty model is a **shallow** GBT predicting **|residual|**. `SIGMA_FLOOR` prevents
  divide-by-zero when we later compute `|resid|/σ`.

```python
train_feats = difficulty_features(df, supp_train)
sigma_oof = np.full(len(y), np.nan)
for tr, te in gkf.split(train_feats, oof_resid, groups):
    gb = _fit_sigma(train_feats.iloc[tr], oof_resid[tr]); sigma_oof[te] = gb.predict(train_feats.iloc[te])
sigma_oof = np.clip(sigma_oof, SIGMA_FLOOR, None)
sigma_model = _fit_sigma(train_feats, oof_resid)      # full-data sigma for scoring candidates
```
- **σ is itself computed out-of-fold** (group CV) — each row's σ comes from a model that never saw its
  well, so the calibration isn't optimistic. Then a full-data `sigma_model` is fit for candidate scoring.

The conformal calibration:
```python
def conformal_quantile(scores, level):
    n = len(scores); k = min(np.ceil((n+1)*level)/n, 1.0)
    return np.quantile(scores, k)
norm_scores = np.abs(oof_resid) / sigma_oof
q_norm  = {lv: conformal_quantile(norm_scores, lv)       for lv in (0.80, 0.90)}
q_fixed = {lv: conformal_quantile(np.abs(oof_resid), lv) for lv in (0.80, 0.90)}
```
- **Nonconformity score = `|residual| / σ`.** The conformal quantile uses the finite-sample correction
  `⌈(n+1)·level⌉/n` (that's what makes coverage *guaranteed*, not just asymptotic). `q_fixed` is a
  fixed-width comparator (no σ normalization) to prove normalization adds value.

```python
CONF_LEVEL = 0.80
surrogate_sigma = sigma_oof
surrogate_halfwidth = q_norm[CONF_LEVEL] * sigma_oof
surrogate_lower = surrogate_oof - surrogate_halfwidth
surrogate_upper = surrogate_oof + surrogate_halfwidth
```
- Per-row 80% interval = **prediction ± (conformal multiplier × σ)**. Wide where σ is large.

Evaluation:
```python
for lv in (0.80, 0.90):
    ... cov_n = mean(y within normalized interval); w_n = mean width
    ... cov_f = fixed-width coverage; w_f = 2*q_fixed
```
- **Coverage check:** 80%→**0.803**, 90%→**0.903** empirical. Both methods hit marginal coverage, but…

```python
diag['width80'] = 2*q_norm[0.80]*sigma_oof; diag['covered80'] = ...
print(diag.groupby('event_quality_label').agg(...))       # width & coverage by quality
action_bin = pd.cut(df['action_delta_gas_e3m3d'].abs(), [0,4,8,100], labels=['small','medium','large'])
print(diag.assign(action=action_bin).groupby('action', observed=True).agg(...))
```
- **Adaptivity check** — the reason to prefer normalized: **contaminated events get ~11.7 m³/d width vs
  ~8.0 for clean.** The interval *widens where the model is genuinely less reliable*. Fixed-width can't.
- The calibration plot (nominal vs empirical) + the "half-width grows with |residual|" scatter visualize
  exactly this.

**questions**
- *"Why conformal instead of a Bayesian posterior or bootstrap?"* → distribution-free finite-sample
  coverage guarantee, no distributional assumptions, cheap, and it reuses the OOF residuals I already
  have. Perfect for a small noisy dataset.
- *"What does 'normalized' buy you over fixed-width?"* → same marginal coverage, but **adaptive**
  width — narrow where confident, wide where not. That's what makes it a decision tool: two candidates
  with the same mean can get different recommendations because their intervals differ.
- *"Why calibrate on group OOF residuals?"* → so the interval reflects new-well deployment error. Calibrating
  on in-fold residuals would be optimistically narrow.
- *"Limitation?"* → conformal guarantees *marginal* coverage, not *conditional*. Per-tier coverage is
  imperfect (clean over-covered 0.82, low-quality under 0.78, contaminated under 0.75) because features only weakly explain error
  size; Mondrian/group-conditional conformal would fix it with more data.

---

## Cell 21 — Diagnostics (where it works and fails)

```python
resid = y.values - surrogate_oof                 # group OOF residuals = conservative estimate
heldout = df.copy(); heldout['pred'] = surrogate_oof; heldout['residual'] = resid
def _mae(a): return np.mean(np.abs(a))
def _rmse(a): return np.sqrt(np.mean(a**2))
```
- All six panels use **group OOF** (new-well) predictions — the honest, conservative view.

The six panels:
1. **Predicted vs actual**, colored by quality, with the y=x line. Shows R²=0.418, MAE≈2.52, and that
   contaminated points scatter most.
2. **Residual histogram** — symmetric, mean ≈ 0 → **near-unbiased**.
3. **MAE by well** (`barh`, sorted) vs the overall dashed line — surfaces the worst wells (WELL_09/01/11).
4. **Residual boxplots by lift mode** — checks for systematic per-mode bias.
5. **Error by event quality** (MAE & RMSE bars) — worse on noisy labels, as expected.
6. **MAE by gas-action size** (`pd.cut` into small/medium/large) — does error grow with action magnitude?

```python
print(f'  random OOF MAE = {_mae(y.values-surrogate_oof_random):.3f}   group OOF MAE = {_mae(resid):.3f}')
print('Worst 3 wells by MAE:', list(w.tail(3).round(2).items()))
```
- **The money line:** random-OOF MAE ≈ group-OOF MAE (2.53 vs 2.52) → **the model transfers to wells with
  no history.** That near-equality is the strongest evidence the model isn't memorizing wells.

**Likely questions**
- *"How do you know it's not overfitting?"* → group≈random OOF, residuals unbiased and symmetric, and the
  model has 4 features / ~6 coefficients on 320 rows. Boosting/RF don't beat it.
- *"Where does it fail?"* → noisiest lift modes and highest-error wells; contaminated events; anything
  moving only afterflow/close-time (which it correctly scores ≈0, a real miss only if those matter in
  the field); can't represent gas-response saturation at extreme deltas.

---

## Cell 23 — Score candidate actions (the deliverable)

```python
score_X = candidate_actions.rename(columns={
    'candidate_delta_gas_e3m3d':'action_delta_gas_e3m3d',
    'candidate_delta_afterflow_min':'action_delta_afterflow_min',
    'candidate_delta_close_time_min':'action_delta_close_time_min'})
score_X_model = score_X[[c for c in feature_cols if c in score_X.columns]].copy()
```
- **Schema alignment:** candidate columns are named `candidate_*`; rename to the training `action_*`
  names so the fitted surrogate accepts them. Then subset to the model's feature columns.

```python
pred      = surrogate.predict(score_X_model)
supp_cand = support_distance(score_X, is_train=False)
sig_cand  = np.clip(sigma_model.predict(difficulty_features(score_X, supp_cand)), SIGMA_FLOOR, None)
lcb = pred - q_norm[0.80]*sig_cand
ucb = pred + q_norm[0.80]*sig_cand
```
- Reuses **the exact same machinery** as training: surrogate mean, support distance, σ model, and the
  **same conformal multiplier `q_norm[0.80]`**. `lcb`/`ucb` = 80% lower/upper confidence bounds.

The four risk gates (all from decision-time info only):
```python
GAS_P95 = df['action_delta_gas_e3m3d'].abs().quantile(0.95)          # ≈16.1 e3m3d extrapolation cap
out_support = supp_cand > SUPP_P95
large_gas   = score_X['action_delta_gas_e3m3d'].abs().values > GAS_P95
not_clean   = (candidate_actions['downtime_flag']==1) | (candidate_actions['facility_constraint_flag']==1)
low_quality = candidate_actions['data_quality_score'] < 0.60
```

The recommendation rule:
```python
LCB_THRESHOLD = 0.0
recommend = (lcb > LCB_THRESHOLD) & (~out_support) & (~large_gas) & (~not_clean) & (~low_quality)
```
- **Recommend only when EVERY gate passes:**
  1. **`lcb > 0`** — the *entire* 80% interval is above zero → confident gain, not a hopeful mean. This
     ties the decision to uncertainty and needs **no arbitrary gain threshold**.
  2. **in support** — not an extrapolation in feature space.
  3. **`|gas| ≤ p95`** — don't extrapolate the linear response past where it was fit (≈16.1).
  4. **clean regime** — downtime/facility are confounds, never levers, so we don't lean on them.
  5. **data quality ≥ 0.60** — trust the inputs.

```python
def _reason(i):
    if recommend[i]:            return 'recommend'
    if not_clean[i]:            return 'hold: not clean regime'
    if low_quality[i]:          return 'hold: low data quality'
    if out_support[i]:          return 'hold: out of support'
    if large_gas[i]:            return 'hold: gas change beyond p95'
    if lcb[i] <= LCB_THRESHOLD: return 'hold: gain not confident (80% interval crosses 0)'
    return 'hold'
```
- **Human-readable audit trail** — every hold gets a specific reason. Operationally essential and a
  great thing to show a domain expert.

```python
scored['pred_delta_liquid_m3d']=pred.round(3); scored['lcb80']=lcb.round(3); scored['ucb80']=ucb.round(3)
scored['interval_width']=(ucb-lcb).round(3); scored['support_dist']=supp_cand.round(3)
scored['in_support']=~out_support; scored['large_gas_change']=large_gas
scored['recommendation']=['recommend' if r else 'hold' for r in recommend]
scored['reason']=[_reason(i) for i in range(len(scored))]
...
scored.to_csv('scored_candidate_actions.csv', index=False)
```
- Builds the output table (prediction, interval, support, flags, decision, reason) and writes
  `scored_candidate_actions.csv`. **Result: 12 recommended, 120 held** — the recommendations concentrate
  on the **largest confident-positive gas changes (~+14 e3m3d)** on the best-supported, lowest-uncertainty
  wells.

**Likely questions**
- *"Why gate on the lower bound instead of the mean?"* → the brief explicitly says don't argmax the mean.
  LCB>0 means I'm confident (at 80%) the action *helps*, not just that its expected value is positive. It
  makes uncertainty part of the decision and needs no magic gain constant.
- *"Two candidates, same predicted mean, different decisions — bug?"* → intended. Their intervals differ;
  the one with tighter uncertainty can clear LCB>0 while the noisier one can't.
- *"Why exclude the downtime/facility flags as levers?"* → they're confounders, not controllable actions,
  and all candidates are already in the clean regime. Recommending based on them would be acting on an
  association I can't control.
- *"What if a great action is just beyond p95 gas?"* → held by design — the linear model isn't validated
  there. In the field I'd stage it: apply a smaller within-support change, observe, recalibrate, then extend.

---

## Cell 25 — Written summary
The markdown answers the 7 required questions (model, validation, performance, uncertainty, failure modes,
recommendation rule, field readiness). Key field-readiness asks to remember: **interventional/randomized data** (to get causal not associational
effects), **more wells**, **coverage of the afterflow/close-time levers**, a **monitoring + recalibration
loop**, **engineering-owned guardrails**, and a **human-in-the-loop** (this is one input, not an autonomous
controller).

---

## The 8 questions — one-line answers

1. **Why linear over boosting?** Signal is low + near-linear; on group CV nothing beats it; parsimony
   generalizes and is auditable for a safety decision.
2. **How do you avoid leakage?** Pipeline fit inside each fold; post-hoc `event_quality_label` held out;
   an `assert` guard; group CV so test wells are unseen.
3. **Random vs group CV?** Random = re-optimize known wells (matches candidates); group = new well
   (conservative gate). They're ~equal → universal effect, transfers.
4. **How is uncertainty honest?** Conformal on group-OOF residuals → distribution-free coverage (0.803 at
   80%); normalized by σ(x) so width adapts (contaminated 11.7 vs clean 8.0).
5. **Why LCB-based recommendation?** Confidence of gain, not hope in a mean; no arbitrary threshold;
   uncertainty drives the decision.
6. **What's confounding and how handled?** downtime/facility move both response and action; included to
   de-confound the fit, excluded as levers, candidates restricted to clean regime.
7. **Biggest weakness?** Observational label → associational not causal; afterflow/close-time unmodeled;
   conditional coverage imperfect; can't capture saturation.
8. **What before field use?** Interventional/randomized rollout, more wells, lever coverage, monitoring +
   recalibration, engineering guardrails, human-in-the-loop.

---

## Two honest "gotcha"

- **"Your R² is only 0.41 — isn't that weak?"** → Even at near-zero gas action the label still spreads
  ~3.2 m³/d, which implies an achievable R² ceiling near 0.5 — and that 3.2 is an *upper* bound on the noise
  (the group still holds explainable well/mode variation), so the ceiling is at least that. At 0.42 I'm
  approaching the signal ceiling, not underfitting; the headroom is modest and would need better features,
  not a bigger model. A model reporting 0.8 here would be leaking or memorizing.
- **"You call it a surrogate but never do Bayesian optimization."** → Correct, and the brief said not to.
  The surrogate + calibrated uncertainty is exactly the `f(s,x)` + posterior an acquisition function
  would call; I built the piece that has to be trustworthy first, and made the recommendation rule the
  safe, uncertainty-aware stand-in for an acquisition step.
