# Phase 6 Specialized Best-Params Evaluation

This report evaluates the best Phase 6 social-regularized CMF parameters from the Optuna search against a plain baseline CMF on the same filtered warm split. This is separate from the smoke-test report: the smoke tests validated integration and small-grid behavior, while this section uses the specialized Optuna-selected setting for `movielens` / `exponential` / network `0`.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `movielens` |
| Diffusion model | `exponential` |
| Network index | `0` |
| Search result | `data/movielens/social_hyperparam_search_results.json` |
| Max ratings | `5000` |
| Train ratings | `4000` |
| Warm test ratings | `643` |
| Random state | `42` |
| L-BFGS iterations | `25` |
| Social retries | `8` |
| User attributes | enabled for social CMF |

## Optuna Best Parameters

| Parameter | Value |
| --- | ---: |
| `k` | `33` |
| `lambda_reg` | `1.53891327083` |
| `w_main` | `0.898452889022` |
| `w_user` | `0.04967517655` |
| `lambda_social` | `0.000198628235669` |
| `social_mode` | `boundary_downweight` |
| `beta` | `0.778428375842` |
| `gamma` | `2.54773196403` |

## Social Edge Summary

| Metric | Value |
| --- | ---: |
| Edges | `928` |
| Mean weight | `0.999999880791` |
| Min weight | `0.312930107117` |
| Max weight | `1.26408565044` |

## Baseline vs Social CMF

The baseline is a plain CMF model without side-user attributes and without the social Laplacian. It uses the same `k` and `lambda_reg` as the best social model so this comparison isolates the added Phase 6 ingredients on the same train/test split.

| Model | User attributes | Social regularizer | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| Plain baseline CMF | no | no | `0.865555` | `0.682517` | `0.196543` |
| Social CMF best params | yes | yes | `0.857745` | `0.672042` | `0.210976` |

| Delta | Value |
| --- | ---: |
| RMSE delta | `-0.007810` |
| MAE delta | `-0.010475` |
| R2 delta | `0.014434` |
| Relative RMSE improvement | `0.902%` |

## Diagnostics

| Diagnostic | Value |
| --- | --- |
| Baseline rating-scale sane | `true` |
| Social rating-scale sane | `true` |
| Social selected attempt | `1` |

## Interpretation

The Optuna-selected Phase 6 model produced a rating-scale sane fit on this rerun. It improves RMSE and R2 over the plain CMF baseline on the same filtered warm split. Because the baseline is intentionally held to the same `k` and `lambda_reg`, the comparison reads as an ablation of user-side features plus the social Laplacian rather than a contest against a separately tuned baseline.

This result is strong enough to carry the selected parameter region into the later full analysis across all sampled networks and diffusion models. The next full-analysis step should re-evaluate these settings across the network grid rather than treating the single network-index result as final model evidence.

## Artifacts

```text
data/movielens/social_best_params_eval_results.json
docs/reports/social_best_params_movielens_exponential.md
```
