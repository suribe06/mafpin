# Core Experiment Plan: Phase 6 Main Evaluation

## Status and Scope

Phase 6 can now be treated as **code-complete for the main experiment**: the patched social-regularized CMF path is implemented, wired through the pipeline, covered by smoke tests, and instrumented with artifact checks and normalization metadata. The smoke tests are good enough to proceed.

Do **not** describe Phase 6 as scientifically complete yet. The correct phrasing is:

> Phase 6 is implemented and smoke-tested end to end; the remaining work is the preregistered main experiment that decides whether the added boundary attributes and social regularization improve recommendation quality on held-out data.

The main experiment should answer three questions:

1. Does the enhanced CMF improve over a normal CMF baseline?
2. Does boundary-guided social regularization improve over enhanced CMF with user attributes alone?
3. Which inferred network family and alpha value should the recommender select for a dataset?

## Core Model Ladder

Use a ladder where each model adds exactly one methodological layer. This is the cleanest way to explain attribution.

| ID | Model | Purpose |
| --- | --- | --- |
| M1 | Baseline CMF | Plain user-item CMF without inferred-network features. This is the primary baseline. |
| M2 | Enhanced CMF, centrality-only attributes | Tests the ws-dmaa-style contribution from inferred-network topology. |
| M3 | Enhanced CMF, centrality + community/boundary attributes | Tests whether overlapping-community and boundary signals add value as side information. |
| M4a | Social CMF with uniform social weights | Tests whether the patched graph regularizer itself is useful, independent of boundary logic. |
| M4b | Social CMF with community-jaccard weights | Tests community-aware smoothing without boundary downweighting. |
| M4c | Social CMF with boundary-downweight weights | Main Phase 6 model: boundary-aware adaptive social regularization. |
| M4d | Social CMF with bridge-preserve weights | More flexible boundary-preserving variant; keep as secondary unless it clearly wins. |

The headline comparison should be:

```text
M4c: Enhanced CMF + boundary attributes + boundary-guided social regularization
vs.
M1: Baseline CMF
```

The important ablation comparisons are:

```text
M2 vs M1  -> value of inferred-network centrality attributes
M3 vs M2  -> value of overlapping-community / boundary attributes
M4a vs M3 -> value of adding any graph regularizer
M4c vs M4a -> value of boundary-aware weighting over uniform smoothing
M4c vs M4b -> value of boundary downweighting beyond community overlap
M4d vs M4c -> whether bridge preservation is worth the extra flexibility
```

For paper/report language, avoid saying that M4c alone proves the boundary layer works. The boundary claim needs `M3 vs M2` and `M4c vs M4a/M4b`.

## Network and Alpha Selection Problem

The recommender needs to choose not only CMF hyperparameters, but also a representation of the inferred user network. In this project that representation is:

```text
network choice = (diffusion model, alpha index)
```

where the diffusion model is one of:

```text
exponential, powerlaw, rayleigh
```

and alpha index is the position in the saved NetInf alpha grid.

The current inference grid is the right default for the main experiment:

| Diffusion model | Alpha grid |
| --- | --- |
| `exponential` | 100 log-spaced values from `alpha_center / 100` to `alpha_center * 100`; `alpha_center` is derived from the median cascade inter-event delta. |
| `rayleigh` | 100 log-spaced values from `alpha_center / 100` to `alpha_center * 100`; `alpha_center` is derived from the median cascade inter-event delta. |
| `powerlaw` | 100 linearly spaced values from `1.1` to `5.0`. |

This gives 300 candidate networks per dataset. Use all 300 for the final network-selection pass when compute allows.

Do not tune alpha on the test set. Treat `(model, alpha_index)` as a hyperparameter selected on validation data, then report final metrics once on the held-out test set.

## Recommended Selection Protocol

Use a two-stage protocol to control compute while still being fair.

### Stage A: Hyperparameter Search on a Representative Network

For each dataset, use a representative complete network to tune ordinary CMF parameters. The existing pipeline uses the first available network, usually `exponential` index `000`, for search. That is acceptable for an initial main run, but the more defensible version is:

1. Use the current train split only.
2. Select three representative network candidates per diffusion family: low-alpha, mid-alpha, and high-alpha quantiles.
3. Run a small Optuna search on those candidates.
4. Choose a stable hyperparameter region, not just a single lucky trial.

Minimum candidate set:

```text
alpha indices: 0, 49, 99 for each of exponential, powerlaw, rayleigh
```

Better candidate set:

```text
alpha indices: 0, 10, 25, 49, 74, 89, 99 for each diffusion model
```

Do not use all 300 networks for Optuna from the start; that mixes network selection with model hyperparameter search and makes the compute cost explode.

### Stage B: Network Selection Across All Alphas

After choosing hyperparameters for each model family, evaluate every complete network candidate:

```text
3 diffusion models x 100 alpha values = 300 network candidates per dataset
```

For each candidate, compute validation metrics over the training data only. The selected network is the candidate with the best primary validation metric, subject to sanity and stability filters.

Primary network-selection metric:

```text
validation RMSE
```

Tie-breakers:

```text
1. validation MAE
2. validation R2
3. lower variance across CV folds
4. simpler/stabler social mode, in this order: boundary_downweight, community_jaccard, uniform, bridge_preserve
```

A candidate is invalid if:

```text
metrics are non-finite
RMSE is outside the rating-scale sanity threshold
required network / centrality / community artifacts are missing
surrogate or downstream diagnostics indicate skipped or degenerate fits
```

The selection output should be a small JSON file per dataset:

```json
{
  "dataset": "movielens",
  "selected_model": "rayleigh",
  "selected_alpha_index": 34,
  "selected_alpha": 0.000123,
  "selected_variant": "M4c_boundary_downweight",
  "selection_metric": "validation_rmse",
  "validation_rmse": 0.0,
  "validation_mae": 0.0,
  "validation_r2": 0.0,
  "hyperparameters": {},
  "selection_pool": "all_complete_networks"
}
```

## Final Test Evaluation

Once the network and hyperparameters are selected, freeze them and run final evaluation on the global held-out test split.

Report at least:

```text
RMSE
MAE
R2
NDCG@10
Precision@10
Recall@10
MRR
```

The final table should compare M1 through M4d on the same split. The main paper table can be compact:

| Dataset | Model | Network model | Alpha index | RMSE | MAE | R2 | NDCG@10 | Precision@10 | Recall@10 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MovieLens | M1 Baseline CMF | none | none | | | | | | | |
| MovieLens | M2 Centrality CMF | selected | selected | | | | | | | |
| MovieLens | M3 Boundary-attribute CMF | selected | selected | | | | | | | |
| MovieLens | M4a Uniform Social CMF | selected | selected | | | | | | | |
| MovieLens | M4b Community Social CMF | selected | selected | | | | | | | |
| MovieLens | M4c Boundary Social CMF | selected | selected | | | | | | | |
| MovieLens | M4d Bridge Social CMF | selected | selected | | | | | | | |

Also report deltas against M1 and M3:

```text
Delta vs M1: absolute and percent improvement over normal CMF
Delta vs M3: value added by social regularization beyond user attributes
```

## What Should Be Tuned

### Baseline CMF M1

Tune independently:

```text
k: 5 to 50
lambda_reg: 0.01 to 10.0, log scale
```

Do not reuse enhanced/social hyperparameters for M1. A fair baseline gets its own search.

### Enhanced CMF M2/M3

Tune independently:

```text
k: 5 to 50
lambda_reg: 0.01 to 10.0, log scale
w_main: 0.1 to 1.0
w_user: 0.01 to 1.0, log scale
```

M2 and M3 can share the same search procedure, but they should not silently share the exact same winning parameters unless the report says so.

### Social CMF M4a-M4d

Tune:

```text
k: 5 to 50
lambda_reg: 0.01 to 10.0, log scale
w_main: 0.1 to 1.0
w_user: 0.01 to 1.0, log scale
lambda_social: 1e-5 to 1.0, log scale
social_mode: fixed per variant, or categorical only in an exploratory run
beta: 0.0 to 1.0 for boundary_downweight and bridge_preserve
gamma: 0.1 to 3.0 for bridge_preserve
social_normalization: fixed during primary runs
```

Use `mean_weight` as the primary social normalization because it is the current default and the smoke tests show it is stable. Keep `normalized_laplacian`, `sum_weight`, and `n_edges` as secondary robustness runs. Do not mix normalization into the main Optuna search unless compute is abundant; it changes the scale of `lambda_social` and makes the search harder to interpret.

## Should Search Cover All Network Types and All Alphas?

Yes, but not at every stage.

Use this rule:

```text
Hyperparameter search: representative alpha subset.
Network selection: all complete alphas across all three diffusion models.
Final test evaluation: selected network(s), plus robustness summary across all networks.
```

Why not tune everything jointly over all 300 networks from the beginning?

1. It is expensive.
2. It leaks too much model-selection flexibility into a single search.
3. It makes it unclear whether improvement came from CMF parameters, social parameters, or lucky network choice.

Why still evaluate all 300 networks?

Because the recommender must recommend a network type and alpha for the dataset. That is a model-selection problem, and all candidate networks should have a chance during validation.

## Recommended Main Commands

Assuming cascades, inference, communities, and centrality already exist:

### Baseline/enhanced run over all networks

```bash
conda run -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1
```

Repeat for Ciao:

```bash
conda run -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1
```

### Social regularized run over all complete networks

Primary run:

```bash
conda run -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --all-networks \
  --social-regularization \
  --social-normalization mean_weight \
  --social-n-trials 200 \
  --social-search-max-ratings 0 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1
```

Repeat for Ciao:

```bash
conda run -n mafpin python pipeline.py \
  --steps recommend \
  --dataset ciao \
  --all-networks \
  --social-regularization \
  --social-normalization mean_weight \
  --social-n-trials 200 \
  --social-search-max-ratings 0 \
  --cmf-method lbfgs \
  --cmf-maxiter 25 \
  --n-jobs 1
```

If runtime becomes too high, first run with a preregistered subset:

```bash
conda run -n mafpin python pipeline.py \
  --steps recommend \
  --dataset movielens \
  --sample-networks 30 \
  --social-regularization \
  --social-normalization mean_weight \
  --social-n-trials 100 \
  --cmf-method lbfgs \
  --cmf-maxiter 25
```

Then promote only the final candidate model to `--all-networks`.

## Alpha Grid to Use

Use the existing generated alpha grid for the main experiment. Do not invent a new grid unless the current one is missing artifacts.

The canonical grid is:

```text
n_alphas = 100
range factor r = 100
edge budget k = 2 x number of users present in cascades
```

For exponential and Rayleigh:

$$
\alpha_i = 10^{\log_{10}(\alpha_0 / 100) + i \cdot \frac{\log_{10}(\alpha_0 \cdot 100) - \log_{10}(\alpha_0 / 100)}{99}}
$$

for $i = 0, \ldots, 99$.

For power-law:

$$
\alpha_i = 1.1 + i \cdot \frac{5.0 - 1.1}{99}
$$

for $i = 0, \ldots, 99$.

This matches the project artifacts and the article text. If a dataset has missing network/community/centrality files for some indices, exclude those candidates from social runs and record the exclusion count.

## How to Recommend a Network Type and Alpha

For each dataset and model variant that uses network information:

1. Evaluate all complete `(diffusion_model, alpha_index)` candidates on validation/CV metrics.
2. Filter invalid candidates using the sanity checks.
3. Rank by mean validation RMSE.
4. Break ties with MAE, R2, and fold variance.
5. Save the selected candidate and its alpha value.
6. Refit the model using the full training split and evaluate once on the held-out test split.

The recommender's network recommendation is therefore:

```text
Choose the diffusion model and alpha whose validation performance is best for the target model family, after sanity filtering, and freeze that choice for final test evaluation.
```

For a more robust publication result, report both:

```text
best selected network result
mean +/- std over all valid networks
```

The selected result shows what the system would deploy. The mean/std shows that the method is not dependent on one lucky alpha.

## Outputs to Save

Per dataset, save:

```text
data/<dataset>/baseline_search_results.json
data/<dataset>/enhanced_search_results.json
data/<dataset>/social_hyperparam_search_results.json
data/<dataset>/network_selection_results.json
data/<dataset>/core_experiment_results.csv
data/<dataset>/core_experiment_by_network.csv
data/<dataset>/core_experiment_summary.md
```

Minimum columns for `core_experiment_by_network.csv`:

```text
dataset
model_variant
diffusion_model
alpha_index
alpha_value
network_edges
k
lambda_reg
w_main
w_user
lambda_social
social_mode
social_normalization
beta
gamma
rmse
mae
r2
ndcg_at_10
precision_at_10
recall_at_10
mrr
rmse_delta_vs_baseline
rmse_delta_vs_m3
valid_metric_row
invalid_reason
```

## Interpretation Rules

Use these rules before writing claims:

1. M4c must beat M1 to claim the full Phase 6 recommender improves over normal CMF.
2. M3 must beat M2 to claim boundary attributes help as side information.
3. M4c must beat M4a or M4b to claim boundary-aware social weighting helps beyond generic social smoothing.
4. If M4a beats M4c, the social regularizer helps, but the boundary weighting design needs revision.
5. If M3 beats M4c, use boundary attributes but do not claim social regularization is beneficial.
6. If results differ by dataset, report that as a dataset interaction, not as a failure.

## Practical Recommendation for the First Main Run

Start with this disciplined sequence:

1. Run MovieLens M1/M3/M4c on all networks.
2. Run Ciao M1/M3/M4c on all networks.
3. Inspect invalid-fit rate and selected `(model, alpha)`.
4. Add M2, M4a, M4b, and M4d ablations.
5. Repeat the best model with `normalized_laplacian` as a robustness check.
6. Only then run SHAP for the selected final models.

This keeps the first main result focused: normal CMF vs enhanced CMF with user attributes and boundary-guided social regularization, while preserving the ablations needed to explain where the improvement comes from.
