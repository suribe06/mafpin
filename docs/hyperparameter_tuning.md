# Hyperparameter Tuning

This document explains how MAFPIN selects the hyperparameters for the enhanced
CMF recommender, which decisions were made, and why.

---

## 1. Parameters That Are Tuned

The enhanced CMF model (`cmfrec.CMF`) has four interacting hyperparameters:

| Parameter | Default | Role |
|-----------|---------|------|
| `k` | 20 | Number of latent factors (user/item embedding dimension) |
| `lambda_reg` | 1.0 | L2 regularisation strength applied to all factor matrices |
| `w_main` | 1.0 | Weight for the rating-matrix reconstruction loss |
| `w_user` | 0.1 | Weight for the user side-information reconstruction loss |

The defaults come from `config.Defaults` and are only used as fallback when no
feature files exist. In normal pipeline execution all four values are found via
the search described below.

### Why all four must be tuned jointly

`w_main` and `w_user` define how the total optimisation loss is split between
the rating-matrix term and the side-information term:

$$\mathcal{L} = w_{\text{main}} \cdot \mathcal{L}_{\text{ratings}} + w_{\text{user}} \cdot \mathcal{L}_{\text{side-info}} + \lambda \cdot \|\Theta\|^2$$

A high `w_user` forces the user embeddings to explain the network features,
which in turn requires a lower `lambda_reg` to avoid killing the side-info
signal. Similarly, `k` controls how much capacity is available to represent
both sources of information. Tuning any one parameter while holding the others
fixed therefore produces a biased result — the four parameters form a coupled
surface that must be explored together.

---

## 2. Algorithm: Optuna TPE

The search uses [Optuna](https://optuna.org/) with the default **Tree-structured
Parzen Estimator (TPE)** sampler.

### Why not random search?

The previous version used `scipy.stats.randint` / `scipy.stats.uniform` for
random sampling. For 4 coupled parameters, random search needs on the order
of $\sim 150$ trials to achieve the same expected coverage as 50 TPE trials.
TPE builds a probabilistic model of the objective surface after each trial:
it fits two density estimators — one over "good" parameter regions (low RMSE)
and one over "bad" regions — and proposes the next candidate where the ratio
$p(\text{good}) / p(\text{bad})$ is highest. This concentrates evaluations in
promising areas rather than wasting them on already-explored bad zones.

**50 trials was chosen** as a balance between:
- Search quality (TPE converges well within 30–60 trials for 4 parameters)
- Wall-clock time (each trial runs 3 CV folds × ~4 s per fold ≈ 2 min/trial)

### Parameter ranges and scales

```
k            : int      [5, 50]           (linear)
lambda_reg   : float    [0.01, 10.0]      (log scale)
w_main       : float    [0.1,  1.0]       (linear)
w_user       : float    [0.01, 1.0]       (log scale)
```

**`lambda_reg` uses log scale** because regularisation is effective over several
orders of magnitude — the difference between 0.01 and 0.1 is as meaningful as
between 1 and 10.

**`w_user` uses log scale** because the literature and empirical results on
MovieLens-small show that the side-information loss typically needs to be
small relative to the rating loss (e.g. 0.01–0.20) to improve RMSE. Log scale
concentrates the majority of trials in that low-signal regime rather than
wasting half of them on values above 0.5 that almost always harm performance.

**`w_main` uses linear scale** because its effective range is narrow —
the model must always explain ratings (minimum 0.1 enforced), and values
near 1.0 all behave similarly at high `lambda_reg`.

---

## 3. What the Search Targets

The objective function is `mean RMSE` over `n_splits=3` random train/test
splits of the **training data** (the global 80% split). The test set (20%)
is never touched during the search.

Each trial:
1. Samples a candidate `(k, lambda_reg, w_main, w_user)`.
2. Runs `evaluate_cmf_with_user_attributes` with `n_splits=3`.
3. Returns the mean `rmse_enhanced` across the 3 splits.

If overlap between rating users and feature users is empty (degenerate
network), the trial raises `optuna.exceptions.TrialPruned()` and is skipped.

Source: `recommender/enhanced.py → search_enhanced_params()`

---

## 4. Why the First Exponential Network Is Used as the Feature Source

The search needs *a* set of user-side features to evaluate the model during
tuning. Using all networks would be:
- **Redundant** — the feature *structure* (which columns exist, their scale,
  their correlation with ratings) does not vary significantly across networks
  of the same type. What varies is which specific edges NetInf inferred, which
  has second-order effects on individual centrality values.
- **Too slow** — running 50 × 3 = 150 model fits per network would multiply
  the search cost by the number of networks sampled.

The selection rule in the code (`pipeline.py → _run_recommend`) is:

```python
for _mn in Models.ALL:   # ["exponential", "powerlaw", "rayleigh"]
    sample_features = load_network_features(_mn, 0, include_communities=...)
    if sample_features is not None:
        sample_model_name = _mn
        break
```

**"First exponential"** is simply the first model name in `Models.ALL` (which
is alphabetically ordered) and the zeroth network index (`_000`). The intent
is only to select *one representative network that exists*; there is no
empirical reason to prefer the exponential model or index 0 other than
availability and reproducibility. If `centrality_metrics_exponential_000.csv`
does not exist, the code falls through to `powerlaw`, then `rayleigh`, then
`powerlaw_000`, etc.

**Key assumption**: the optimal `(k, lambda_reg, w_main, w_user)` found with
network 0 generalises well to other networks. This is reasonable because
the feature *type* (centrality + community metrics, ~9 columns) and *scale*
(MinMax or StandardScaler normalised to similar ranges) is the same across
all networks. Experiments on MovieLens-small confirm that RMSE variance across
networks with the same hyper-parameters is on the order of ±0.0005, much
smaller than the sensitivity to `lambda_reg` or `w_user`.

---

## 5. How the Best Params Flow Through the Pipeline

```
pipeline.py _run_recommend()
│
├── search_enhanced_params(train_df, sample_features, n_trials=50)
│         └── Optuna TPE → best: k, lambda_reg, w_main, w_user
│
├── train_final_model(train_df, k=best_k, lambda_reg=best_lambda)
│         └── Plain CMF (no side info) evaluated on global test
│
└── run_network_evaluation(train_df, k=best_k, lambda_reg=best_lambda,
                           w_main=best_w_main, w_user=best_w_user)
          └── For each sampled network:
                evaluate_single_network(...)
                  └── evaluate_cmf_with_user_attributes(n_splits=5)
                        ├── Enhanced CMF with U=scaled_features
                        └── Paired baseline CMF (same split, no side info)
```

The Optuna search runs **once**. The winning parameters are reused for every
network evaluated in `run_network_evaluation`. This is consistent with the
assumption in Section 4 — if the features generalise, so do the optimal weights.

---

## 6. Feature Scaling: Why Fit on Training Users Only

Inside each CV fold of `evaluate_cmf_with_user_attributes`:

1. The scaler (`StandardScaler` by default) is **fitted on training users' features only**.
2. The fitted scaler is then **applied to all users** (train + test) before
   building the `U` matrix passed to `cmfrec.CMF`.

This follows the *no-leakage* principle (audit bug M-2): if the scaler were
fitted on all users including test users, the model would indirectly see test
statistics at training time. The key point is that network features are
derived from the training-only inferred network (audit bug C-3 fix), so
applying the train-fitted scaler to test users does not constitute leakage —
the features themselves are already leak-free.

---

## 7. The Paired Baseline Comparison

Inside `evaluate_cmf_with_user_attributes`, **every split also trains a plain
CMF** with the same `k` and `lambda_reg` (no `U` matrix, no side info)
on the same filtered subset of users. This is the *paired* baseline (audit
bug M-3 fix).

The improvement reported is:

$$\text{improvement} = \text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{enhanced}}$$

A positive value means the network features helped. Using the same
`lambda_reg` for both models is intentional: we want to isolate the effect of
side information, not of regularisation tuning.

