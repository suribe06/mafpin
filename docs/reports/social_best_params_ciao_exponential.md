# Phase 6 Specialized Best-Params Evaluation

This report evaluates the best Phase 6 social-regularized CMF parameters from the Optuna search against a plain baseline CMF on the same filtered warm split. This is separate from the smoke-test report: the smoke tests validated integration and small-grid behavior, while this section uses the specialized Optuna-selected setting for `ciao` / `exponential` / network `0`.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `ciao` |
| Diffusion model | `exponential` |
| Network index | `0` |
| Search result | `data/ciao/social_hyperparam_search_results.json` |
| Max ratings | `5000` |
| Train ratings | `4000` |
| Warm test ratings | `286` |
| Random state | `42` |
| L-BFGS iterations | `25` |
| Social retries | `8` |
| User attributes | enabled for social CMF |

## Optuna Best Parameters

| Parameter | Value |
| --- | ---: |
| `k` | `42` |
| `lambda_reg` | `1.66438097934` |
| `w_main` | `0.609488230159` |
| `w_user` | `0.213415637907` |
| `lambda_social` | `0.0541532114524` |
| `social_mode` | `bridge_preserve` |
| `beta` | `0.854885736263` |
| `gamma` | `0.496106366478` |

## Social Edge Summary

| Metric | Value |
| --- | ---: |
| Edges | `4423` |
| Mean weight | `1` |
| Min weight | `0.612894654274` |
| Max weight | `1.27657341957` |

## Baseline vs Social CMF

The baseline is a plain CMF model without side-user attributes and without the social Laplacian. It uses the same `k` and `lambda_reg` as the best social model so this comparison isolates the added Phase 6 ingredients on the same train/test split.

| Model | User attributes | Social regularizer | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| Plain baseline CMF | no | no | `0.918767` | `0.695166` | `0.160016` |
| Social CMF best params | yes | yes | `0.914449` | `0.698695` | `0.167893` |

| Delta | Value |
| --- | ---: |
| RMSE delta | `-0.004318` |
| MAE delta | `0.003529` |
| R2 delta | `0.007877` |
| Relative RMSE improvement | `0.470%` |

## Diagnostics

| Diagnostic | Value |
| --- | --- |
| Baseline rating-scale sane | `true` |
| Social rating-scale sane | `true` |
| Social selected attempt | `0` |

## Interpretation

The Optuna-selected Phase 6 model produced a rating-scale sane fit on this rerun. It improves RMSE and R2 over the plain CMF baseline on the same filtered warm split. Because the baseline is intentionally held to the same `k` and `lambda_reg`, the comparison reads as an ablation of user-side features plus the social Laplacian rather than a contest against a separately tuned baseline.

This result is strong enough to carry the selected parameter region into the later full analysis across all sampled networks and diffusion models. The next full-analysis step should re-evaluate these settings across the network grid rather than treating the single network-index result as final model evidence.

## Artifacts

```text
data/ciao/social_best_params_eval_results.json
docs/reports/social_best_params_ciao_exponential.md
```
