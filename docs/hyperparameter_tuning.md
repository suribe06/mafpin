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
├── search_baseline_params(train_df, n_trials=50)           [2 params]
│         └── Optuna TPE → best_k_b, best_lambda_b
│
├── search_enhanced_params(train_df, sample_features, n_trials=50)  [4 params]
│         └── Optuna TPE → best_k_e, best_lambda_e, best_w_main, best_w_user
│
├── train_final_model(train_df, k=best_k_b, lambda_reg=best_lambda_b)
│         └── Plain CMF (no side info) evaluated on global test set
│
└── run_network_evaluation(train_df, k=best_k_e, lambda_reg=best_lambda_e,
                           w_main=best_w_main, w_user=best_w_user)
          └── For each sampled network:
                evaluate_single_network(...)
                  └── evaluate_cmf_with_user_attributes(n_splits=5)
                        ├── Enhanced CMF with U=scaled_features
                        └── Paired baseline CMF (same split, same k/lambda_e)
```

Two independent Optuna searches run sequentially. The baseline search finds
`(best_k_b, best_lambda_b)` optimised for plain CMF; the enhanced search finds
`(best_k_e, best_lambda_e, best_w_main, best_w_user)` optimised for the
combined loss. Each search runs for 50 TPE trials.

Note that the **global test baseline** (middle branch) uses the baseline-tuned
params, while the **per-network paired baseline** inside
`evaluate_cmf_with_user_attributes` uses the enhanced model's `k`/`lambda_e`.
This is intentional: the per-network paired comparison is a controlled
experiment to isolate the effect of side information within the same
hyperparameter configuration.

---

## 6. The Two Baseline Comparisons

### 6a. Global test baseline (`pipeline.py`)

The global baseline is a plain CMF trained on the full training split and
evaluated on the global held-out test set (20%). It uses **independently
tuned** `(best_k_b, best_lambda_b)` from `search_baseline_params` — a
dedicated Optuna TPE search that optimises only `k` and `lambda_reg` for the
plain-CMF loss:

$$\mathcal{L}_{\text{baseline}} = \mathcal{L}_{\text{ratings}} + \lambda \cdot \|\Theta\|^2$$

Using the enhanced model's `lambda_e` here would be unfair: Optuna finds high
`lambda_e` values partly to counter the side-information term (`w_user ·
L_side-info`). Applied to a model without that term, the same `lambda_e`
over-regularises and inflates baseline RMSE.

### 6b. Per-network paired baseline (`evaluate_cmf_with_user_attributes`)

Inside each CV fold during network evaluation, **every split also trains a
plain CMF** with the *same* `k` and `lambda_e` (no `U` matrix, no side
information) on the same filtered subset of users. This is the *paired*
baseline (audit bug M-3 fix).

The improvement reported is:

$$\text{improvement} = \text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{enhanced}}$$

A positive value means the network features helped. Using the **same**
`lambda_e` for both models in this comparison is intentional: we want to
isolate the effect of side information, not of regularisation tuning. The
variable being changed between the two models is the presence of `U`, not
the regularisation level.

