# Boundary-Guided Social Regularization

This document describes `recommender.enhanced.social_regularization`, the Phase 6 module that adds a direct social graph penalty to the local C-backed `cmfrec` model.

The implementation is reusable as a standalone module and is also wired into the unified pipeline through `python pipeline.py --social-regularization`. Phases 1-5 continue to use network features as user side-information; Phase 6 additionally uses the inferred user-user network as a regularizer over latent user factors.

## Model Objective

The patched `cmfrec` L-BFGS path adds an edge-wise penalty over user latent vectors:

$$
\lambda_S \sum_{(u,v) \in E} w_{uv}\|p_u - p_v\|^2
$$

where:

| Symbol | Meaning |
| --- | --- |
| $p_u$, $p_v$ | Latent vectors for users `u` and `v` |
| $E$ | Edges from the NetInf-inferred user-user network |
| $w_{uv}$ | Boundary-guided social edge weight |
| $\lambda_S$ | Social regularization strength, passed as `lambda_social` |

The gradient contribution for an edge is:

$$
2\lambda_S w_{uv}(p_u - p_v)
$$

added to user `u`, with the opposite sign added to user `v`. This is equivalent to a weighted graph Laplacian penalty on the latent user matrix.

## Why This Uses L-BFGS

The social penalty couples neighboring users, so the user factors can no longer be optimized row-by-row independently. That makes the existing ALS path a poor fit for Phase 6. The local `cmfrec` L-BFGS objective evaluates all user factors together, so the edge loop can be added directly in C without a slow Python optimization loop.

## Input Artifacts

The module expects the existing pipeline outputs for one dataset, diffusion model, and network index:

| Artifact | Example |
| --- | --- |
| Inferred network | `data/movielens/inferred_networks/exponential/inferred-network-expo-000.txt` |
| Community features | `data/movielens/communities/exponential/communities_exponential_000.csv` |
| Enhanced user features | `data/movielens/centrality_metrics/exponential/centrality_metrics_exponential_000.csv` plus community/cascade joins |
| Ratings | `datasets/movielens/ratings_small.csv` |

The inferred network is loaded with `networks.network_io.load_as_networkx()` and then converted to an undirected graph with `directed_to_undirected(..., method="union")` by default.

## Social Weight Modes

The `social_mode` parameter controls how each edge weight `w_uv` is computed.

| Mode | Formula | Interpretation |
| --- | --- | --- |
| `uniform` | $w_{uv}=1$ | Plain graph smoothing; useful as a solver sanity check. |
| `community_jaccard` | $w_{uv}=J(C(u), C(v))$ | Smooths users more when they share overlapping communities. |
| `boundary_downweight` | $w_{uv}=J(C(u), C(v))\cdot(1-\beta\max(b_u,b_v))$ | Reduces smoothing when either endpoint is boundary-like. This is the default Phase 6 mode. |
| `bridge_preserve` | $w_{uv}=\sigma(\gamma J(C(u), C(v))-\beta\max(b_u,b_v))$ | Uses a sigmoid balance between shared-community agreement and boundary tension. |

Definitions:

| Quantity | Source |
| --- | --- |
| `C(u)` | Set parsed from the `community_ids` column. |
| `J(C(u), C(v))` | Jaccard similarity between endpoint community sets. |
| `b_u` | Boundary intensity in `[0, 1]`; derived from negative `lph_score` when available, otherwise `is_boundary`, otherwise inverted `local_pluralistic_hom`. |
| `beta` | Boundary penalty strength. Larger values reduce smoothing around boundary users. |
| `gamma` | Shared-community reward used by `bridge_preserve`. Larger values increase smoothing for high-overlap users. |

Weights are normalized to mean 1 by default so that `lambda_social` has roughly comparable scale across network indices and weight modes.

## Public API

### `SocialEdges`

Dataclass containing the COO arrays passed to patched `cmfrec`:

| Field | Description |
| --- | --- |
| `row` | `int32` source user IDs. |
| `col` | `int32` target user IDs. |
| `val` | Edge weights as `float32` or `float64`. |
| `mode` | Weighting mode used to build the edges. |
| `n_edges` | Number of usable weighted social edges. |
| `mean_weight`, `min_weight`, `max_weight` | Weight diagnostics after optional normalization. |

### `load_community_frame(dataset, model_name, network_index)`

Loads the matching community CSV and indexes it by `UserId`. This is mainly a helper for social edge construction and diagnostics.

### `build_social_edges(...)`

Builds the weighted social COO arrays:

```python
social_edges = build_social_edges(
    dataset="movielens",
    model_name="exponential",
    network_index=0,
    user_index=user_attributes.index,
    mode="boundary_downweight",
    beta=0.5,
    gamma=1.0,
)
```

Important parameters:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `dataset` | required | Dataset key from `config.Datasets.ALL`. |
| `model_name` | required | Diffusion model: `exponential`, `powerlaw`, or `rayleigh`. |
| `network_index` | required | Zero-based inferred-network index. |
| `user_index` | required | Encoded users eligible for social regularization. |
| `mode` | `boundary_downweight` | Social weight formula. |
| `beta` | `0.5` | Boundary downweight strength. |
| `gamma` | `1.0` | Community-overlap strength for `bridge_preserve`. |
| `symmetrization` | `union` | Directed-to-undirected conversion method. |
| `normalization` | `mean_weight` | Social edge normalization strategy: `mean_weight`, `sum_weight`, `n_edges`, `normalized_laplacian`, or `none`. Legacy aliases `mean` and `edges` map to `mean_weight` and `n_edges`. |

### `fit_social_cmf_split(...)`

Fits one train/test split with the patched local `CMF`:

```python
model, metrics = fit_social_cmf_split(
    train_df,
    test_df,
    user_attributes,
    social_edges,
    k=8,
    lambda_reg=10.0,
    lambda_social=0.1,
    maxiter=20,
    include_user_attributes=False,
)
```

The default solver is `method="lbfgs"`, because the social regularizer is implemented in the L-BFGS objective. When `include_user_attributes=False`, the fit isolates the social graph penalty. When `include_user_attributes=True`, the module also passes the Phase 5 side-user matrix as `U`; this path is available but should be tuned separately.

### `fit_social_cmf_model(...)`

Fits the same social-regularized CMF model without evaluating a train/test split.
This is used by the SHAP pipeline path, where the fitted model is needed so a
surrogate can explain its predictions:

```python
model = fit_social_cmf_model(
    train_df,
    user_attributes,
    social_edges,
    k=20,
    lambda_reg=1.0,
    lambda_social=0.001,
    include_user_attributes=True,
)
```

## Unified Pipeline Usage

Run recommendation, Optuna search, and SHAP with social regularization enabled:

```bash
python pipeline.py --steps recommend shap --social-regularization
```

Or run the full pipeline from cascades through SHAP:

```bash
python pipeline.py --all --social-regularization
```

When this flag is enabled:

1. `recommend` trains the plain baseline with the pipeline CMF method, which defaults to `lbfgs`.
2. The social Optuna search writes `data/<dataset>/social_hyperparam_search_results.json`.
3. Network evaluation calls `build_social_edges(...)` and `fit_social_cmf_split(...)` for each sampled inferred network.
4. `shap` loads the social search result and trains social CMF models before fitting the GBT surrogate.

The most common knobs are:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--social-regularization` | off | Enable the Phase 6 social CMF path. |
| `--cmf-method` | `lbfgs` | CMF optimizer. Social regularization requires `lbfgs`. |
| `--cmf-maxiter` | `25` | L-BFGS iteration budget. |
| `--social-mode` | `boundary_downweight` | Edge weighting mode. |
| `--lambda-social` | `0.001` | Fallback social penalty strength. |
| `--social-beta` | `0.5` | Boundary penalty parameter. |
| `--social-gamma` | `1.0` | Shared-community gain parameter. |
| `--social-normalization` | `mean_weight` | Social edge normalization strategy for social-regularized CMF. |
| `--social-search-max-ratings` | `5000` | Rating cap for social Optuna search. |
| `--social-n-trials` | `200` | Optuna trial budget for the eight-parameter social CMF search. |

## Smoke-Test Runner

The smoke-test entry point lives in `recommender.enhanced.social_smoke_test`, not in this module. That file imports the reusable functions above, runs the before/after fit, and writes `data/<dataset>/social_smoke_results.json`. It accepts the same `--social-normalization` choices as the pipeline so normalization variants can be probed before full dataset reruns.

See `docs/reports/social_smoke_test.md` for the current smoke-test report.
