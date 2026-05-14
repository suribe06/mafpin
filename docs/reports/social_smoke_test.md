# Phase 6 Social Regularization Smoke-Test Report

This report summarizes the first small smoke test for Boundary-Guided Social Regularization. The goal is to verify that the patched local `cmfrec` L-BFGS path, the social edge builder, and the evaluation wrapper work together end to end.

The reusable module is documented separately in `docs/social_regularization.md`. The smoke-test runner is `recommender.enhanced.social_smoke_test`.

## Recommendation

Use this result as an initial integration report without user attributes. The social-regularized run isolates the new Phase 6 regularizer and produced finite, rating-scale metrics on a small warm-only sample.

User attributes should remain a separate follow-up condition. They are still supported by the runner through `--include-user-attributes`, but the combined side-user plus social objective needs its own tuning pass before it should be interpreted.

## Experiment Roadmap

The smoke-test report will be expanded step by step:

1. Run the smoke test over all four social modes with the same network index. Status: completed below.
2. Sweep `lambda_social` over a small log grid such as `0.001`, `0.01`, `0.1`, `1.0`. Status: completed below.
3. Repeat on 10 random network indices for each diffusion model: exponential, powerlaw, and rayleigh. Status: completed below.
4. Add user attributes back with a constrained grid over `w_user` and `lambda_reg`. Status: completed below.
5. Promote the best stable settings into the full study pipeline.

## Command

```bash
conda run -n mafpin python -m recommender.enhanced.social_smoke_test \
  --dataset movielens \
  --model exponential \
  --network-index 0 \
  --social-mode boundary_downweight \
  --lambda-social 0.1 \
  --max-ratings 5000 \
  --k 8 \
  --lambda-reg 10 \
  --maxiter 20 \
  --nthreads 1 \
  --output-path data/movielens/social_smoke_results/boundary_downweight_lambda_0.1.json
```

The four-mode Step 1 run used:

```bash
for mode in uniform community_jaccard boundary_downweight bridge_preserve; do
  conda run -n mafpin python -m recommender.enhanced.social_smoke_test \
    --dataset movielens \
    --model exponential \
    --network-index 0 \
    --social-mode "$mode" \
    --lambda-social 0.1 \
    --max-ratings 5000 \
    --k 8 \
    --lambda-reg 10 \
    --maxiter 20 \
    --nthreads 1 \
    --output-path "data/movielens/social_smoke_results/${mode}_lambda_0.1.json"
done
```

The Step 2 lambda sweep used the same loop structure with `lambda_social` in `0.001`, `0.01`, `0.1`, and `1.0`, writing each run to `data/movielens/social_smoke_results/lambda_sweep/`.

The MovieLens Step 2 plots were generated with:

```bash
python -m visualization.model_plots.social_regularization --dataset movielens
```

The MovieLens Step 3 network sweep used the Step 2 RMSE/R2 winner, `boundary_downweight` with `lambda_social=0.001`, and sampled 10 complete network indices per diffusion model with `random_state=42`:

```bash
conda run -n mafpin python -m recommender.enhanced.social_network_sweep \
  --dataset movielens \
  --models exponential powerlaw rayleigh \
  --n-networks 10 \
  --social-mode boundary_downweight \
  --lambda-social 0.001 \
  --max-ratings 5000 \
  --k 8 \
  --lambda-reg 10 \
  --maxiter 20 \
  --nthreads 1 \
  --random-state 42
```

The MovieLens Step 3 plots were generated with:

```bash
python -m visualization.model_plots.social_regularization --dataset movielens --plot-kind network
```

For Ciao, the same commands were run with `--dataset ciao`; Step 3 again used the Ciao Step 2 winner, `boundary_downweight` with `lambda_social=0.001`.

The Step 4 user-attribute grid used the Step 2/3 social setting, `boundary_downweight` with `lambda_social=0.001`, and swept a constrained grid over `lambda_reg` and `w_user`:

```bash
conda run -n mafpin python -m recommender.enhanced.social_smoke_test \
  --user-attribute-grid \
  --dataset movielens \
  --model exponential \
  --network-index 0 \
  --social-mode boundary_downweight \
  --lambda-social 0.001 \
  --max-ratings 5000 \
  --k 8 \
  --maxiter 20 \
  --nthreads 1 \
  --overwrite
```

For Ciao, the same Step 4 command was run with `--dataset ciao`.

## Dataset: MovieLens

### MovieLens Configuration

| Setting | Value |
| --- | --- |
| Dataset | `movielens` |
| Diffusion model | `exponential` |
| Network index | `0` |
| Social mode | `boundary_downweight` |
| `lambda_social` | `0.1` |
| `beta` / `gamma` | `0.5` / `1.0` |
| `max_ratings` | `5000` |
| Train ratings | `4000` |
| Warm test ratings | `643` |
| Latent factors | `k=8` |
| L2 regularization | `lambda_reg=10.0` |
| L-BFGS iterations | `maxiter=20` |
| User attributes | disabled |

### MovieLens Social Edges

| Metric | Value |
| --- | --- |
| Number of edges | `928` |
| Mean weight | `1.0000001192` |
| Min weight | `0.3147248328` |
| Max weight | `1.2588993311` |

### MovieLens Metrics

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| `lambda_social=0` | `0.874138` | `0.691986` | `0.180528` |
| `lambda_social=0.1` | `0.873996` | `0.691888` | `0.180794` |

The RMSE delta was `-0.000142`, so the social-regularized run was slightly better on this small sample. Treat this as an integration signal rather than evidence of final model performance.

### MovieLens Step 1: Social Mode Comparison

Step 1 ran all four social weighting modes with the same dataset, diffusion model, network index, split seed, latent dimension, regularization, and `lambda_social=0.1`.

Result files:

| Mode | Output file |
| --- | --- |
| `uniform` | `data/movielens/social_smoke_results/uniform_lambda_0.1.json` |
| `community_jaccard` | `data/movielens/social_smoke_results/community_jaccard_lambda_0.1.json` |
| `boundary_downweight` | `data/movielens/social_smoke_results/boundary_downweight_lambda_0.1.json` |
| `bridge_preserve` | `data/movielens/social_smoke_results/bridge_preserve_lambda_0.1.json` |

Social-regularized run metrics:

| Mode | Edges | Weight range | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform` | `1300` | `1.0000` to `1.0000` | `0.873996` | `0.691888` | `0.180794` |
| `community_jaccard` | `928` | `0.3124` to `1.2497` | `0.873997` | `0.691889` | `0.180792` |
| `boundary_downweight` | `928` | `0.3147` to `1.2589` | `0.873996` | `0.691888` | `0.180794` |
| `bridge_preserve` | `1300` | `0.6210` to `1.2025` | `0.873996` | `0.691888` | `0.180795` |

The social-regularized metrics are effectively tied at this smoke-test scale. `bridge_preserve` has the numerically lowest RMSE, but the gap from `uniform` and `boundary_downweight` is below `0.000001`, so it should not be treated as a meaningful ranking yet.

Control-run note: the recomputed `lambda_social=0` control in the `community_jaccard` run produced an unreasonable RMSE despite finite values, while the social-regularized run remained rating-scale sane. For this step, the safest interpretation is the direct comparison among social-regularized runs plus the stable no-social controls from the other modes. The next lambda sweep should keep using saved per-run diagnostics and should avoid overinterpreting any single no-social control fit.

### MovieLens Step 2: Lambda Sweep

Step 2 evaluated all four social weighting modes over this `lambda_social` grid:

```text
0.001, 0.01, 0.1, 1.0
```

Result summary:

| Mode | Best RMSE lambda | Best RMSE | Best MAE lambda | Best MAE | Best R2 lambda | Best R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boundary_downweight` | `0.001` | `0.873873` | `1.0` | `0.691888` | `0.001` | `0.181026` |
| `bridge_preserve` | `0.001` | `0.873933` | `0.001` | `0.691838` | `0.001` | `0.180912` |
| `community_jaccard` | `0.001` | `0.873911` | `0.1` | `0.691889` | `0.001` | `0.180954` |
| `uniform` | `0.001` | `0.873922` | `0.001` | `0.691828` | `0.001` | `0.180934` |

The clearest pattern is that the smallest tested regularization, `lambda_social=0.001`, gives the best RMSE and R2 for every mode. Larger values quickly collapse toward nearly identical scores around `RMSE=0.873996`, `MAE=0.691889`, and `R2=0.18079`. At this smoke-test scale, stronger smoothing appears unnecessary and may slightly flatten useful user-specific variation.

The best RMSE in this sweep is `boundary_downweight` with `lambda_social=0.001` (`RMSE=0.873873`, `R2=0.181026`). The best MAE is `uniform` with `lambda_social=0.001` (`MAE=0.691828`). These differences remain very small, so the correct interpretation is directional rather than definitive: use the next phase to test whether the low-lambda preference survives across networks and diffusion models.

Summary CSV:

```text
data/movielens/social_smoke_results/lambda_sweep/lambda_sweep_summary.csv
```

Plots:

![RMSE over lambda_social](../../plots/movielens/models/social_regularization/social_lambda_rmse.png)

![MAE over lambda_social](../../plots/movielens/models/social_regularization/social_lambda_mae.png)

![R2 over lambda_social](../../plots/movielens/models/social_regularization/social_lambda_r2.png)

Step 2 diagnostic note: one no-social control fit in the lambda sweep again produced unreasonable scale, but every social-regularized fit passed the rating-scale sanity check. The Step 2 plots and table therefore use the social-regularized metrics only.

### MovieLens Step 3: Network Sweep

Step 3 tested whether the low-lambda pattern from Step 2 survives across different inferred social graphs. It fixed `lambda_social=0.001` because Step 2 identified that value as the best RMSE/R2 setting for `boundary_downweight`. In other words, Step 3 is not another lambda sweep; it is a network-robustness check using the best Step 2 configuration.

The sweep used `boundary_downweight`, `lambda_social=0.001`, and 10 random complete network indices for each diffusion model.

Sampled indices:

| Diffusion model | Network indices |
| --- | --- |
| `exponential` | `8`, `9`, `19`, `41`, `60`, `68`, `71`, `82`, `94`, `96` |
| `powerlaw` | `17`, `34`, `39`, `40`, `46`, `54`, `62`, `75`, `81`, `88` |
| `rayleigh` | `6`, `15`, `25`, `34`, `44`, `58`, `66`, `71`, `89`, `95` |

All 30 runs completed successfully and all social-regularized runs passed the rating-scale sanity check.

By-model summary over the 10 sampled networks:

| Diffusion model | Mean RMSE | RMSE std | Mean MAE | MAE std | Mean R2 | R2 std | Mean RMSE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exponential` | `0.873838` | `0.000179` | `0.691838` | `0.000159` | `0.181090` | `0.000336` | `-0.000091` |
| `powerlaw` | `0.873870` | `0.000048` | `0.691934` | `0.000059` | `0.181031` | `0.000090` | `-0.000027` |
| `rayleigh` | `0.873819` | `0.000219` | `0.691853` | `0.000271` | `0.181127` | `0.000410` | `-0.000079` |

The network sweep supports the Step 2 recommendation to keep `lambda_social` small. The average RMSE delta is negative for all three diffusion models, meaning the social-regularized run is slightly better than its no-social control on average. The effect is still very small, but it is stable enough across sampled networks to justify carrying `boundary_downweight` with `lambda_social=0.001` into the next tuning stage.

Among these sampled networks, `rayleigh` has the best mean RMSE and R2, while `exponential` has the best mean MAE. The model-level differences are tiny, so the practical conclusion is not that one diffusion family has clearly won; it is that the Phase 6 regularizer remains numerically stable across all three inferred-network families.

Step 3 artifacts:

```text
data/movielens/social_smoke_results/network_sweep/selected_network_indices.json
data/movielens/social_smoke_results/network_sweep/network_sweep_summary.csv
data/movielens/social_smoke_results/network_sweep/network_sweep_by_model.csv
```

Plots:

![RMSE over sampled networks](../../plots/movielens/models/social_regularization/social_network_rmse.png)

![MAE over sampled networks](../../plots/movielens/models/social_regularization/social_network_mae.png)

![R2 over sampled networks](../../plots/movielens/models/social_regularization/social_network_r2.png)

### MovieLens Step 4: User-Attribute Grid

Step 4 adds user attributes back into the patched social-regularized objective. It fixes the social configuration to `boundary_downweight` with `lambda_social=0.001` and sweeps:

```text
lambda_reg: 1.0, 3.0, 10.0
w_user: 0.01, 0.05, 0.10
```

Result summary:

| `lambda_reg` | `w_user` | RMSE | MAE | R2 | Baseline sane | Social sane |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1.0` | `0.01` | `0.881127` | `0.686927` | `0.167372` | `false` | `true` |
| `1.0` | `0.05` | `0.881396` | `0.686123` | `0.166863` | `false` | `true` |
| `1.0` | `0.10` | `0.884214` | `0.687056` | `0.161528` | `true` | `true` |
| `3.0` | `0.01` | `0.861362` | `0.675527` | `0.204307` | `false` | `true` |
| `3.0` | `0.05` | `0.858254` | `0.672438` | `0.210039` | `false` | `true` |
| `3.0` | `0.10` | `0.860909` | `0.674127` | `0.205144` | `false` | `true` |
| `10.0` | `0.01` | `0.873900` | `0.691947` | `0.180975` | `true` | `true` |
| `10.0` | `0.05` | `0.874075` | `0.692022` | `0.180647` | `false` | `true` |
| `10.0` | `0.10` | `0.873227` | `0.691386` | `0.182236` | `true` | `true` |

The best MovieLens Step 4 social-regularized result is `lambda_reg=3.0`, `w_user=0.05` (`RMSE=0.858254`, `MAE=0.672438`, `R2=0.210039`). This is substantially better than the no-user-attribute Step 3 scale around `RMSE=0.8738` and `R2=0.181`, suggesting that user attributes become useful once regularization is relaxed from `lambda_reg=10.0` to `3.0`.

The no-social controls diverged for several grid cells, while all social-regularized fits remained rating-scale sane. For this smoke-test step, the social metrics are therefore useful for selecting a candidate setting, but invalid no-social controls should not be used as effect-size estimates.

Step 4 artifacts:

```text
data/movielens/social_smoke_results/user_attribute_grid/user_attribute_grid_summary.csv
data/movielens/social_smoke_results/user_attribute_grid/*.json
```

## Dataset: Ciao

### Configuration

| Setting | Value |
| --- | --- |
| Dataset | `ciao` |
| Diffusion model | `exponential` |
| Network index | `0` |
| Social mode | `boundary_downweight` |
| `lambda_social` | `0.1` |
| `beta` / `gamma` | `0.5` / `1.0` |
| `max_ratings` | `5000` |
| Train ratings | `4000` |
| Warm test ratings | `286` |
| Latent factors | `k=8` |
| L2 regularization | `lambda_reg=10.0` |
| L-BFGS iterations | `maxiter=20` |
| User attributes | disabled |

### Social Edges

| Metric | Value |
| --- | --- |
| Number of edges | `2043` |
| Mean weight | `1.0000001192` |
| Min weight | `0.2026623189` |
| Max weight | `1.8109966516` |

### Metrics

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| `lambda_social=0` | `0.933735780` | `0.725957427` | `0.132422559` |
| `lambda_social=0.1` | `0.933735553` | `0.725957313` | `0.132422981` |

The Ciao single-run RMSE delta was `-0.000000227`, so the social-regularized run was again slightly better, but the effect is much smaller than the smoke-test precision needed for a model-quality claim.

### Step 1: Social Mode Comparison

Step 1 ran the same four social weighting modes with the same Ciao dataset split, diffusion model, network index, latent dimension, regularization, and `lambda_social=0.1`.

Social-regularized run metrics:

| Mode | Edges | Weight range | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform` | `4423` | `1.0000` to `1.0000` | `0.933737` | `0.725958` | `0.132420` |
| `community_jaccard` | `2043` | `0.2300` to `1.6103` | `0.933736` | `0.725958` | `0.132422` |
| `boundary_downweight` | `2043` | `0.2027` to `1.8110` | `0.933736` | `0.725957` | `0.132423` |
| `bridge_preserve` | `4423` | `0.6978` to `1.3512` | `0.933735` | `0.725957` | `0.132424` |

At `lambda_social=0.1`, `bridge_preserve` is numerically best on Ciao, but the differences among social modes are in the sixth decimal place or smaller.

### Step 2: Lambda Sweep

Step 2 evaluated all four social weighting modes over the same `lambda_social` grid:

```text
0.001, 0.01, 0.1, 1.0
```

Result summary:

| Mode | Best RMSE lambda | Best RMSE | Best MAE lambda | Best MAE | Best R2 lambda | Best R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boundary_downweight` | `0.001` | `0.933733633` | `0.001` | `0.725955826` | `0.001` | `0.132426550` |
| `bridge_preserve` | `0.01` | `0.933735030` | `0.1` | `0.725957015` | `0.01` | `0.132423953` |
| `community_jaccard` | `1.0` | `0.933734980` | `0.01` | `0.725957249` | `1.0` | `0.132424046` |
| `uniform` | `0.1` | `0.933735529` | `0.001` | `0.725957280` | `0.1` | `0.132423026` |

The best Ciao result for RMSE, MAE, and R2 is `boundary_downweight` with `lambda_social=0.001`. That matches the MovieLens Step 2 winner, so the Ciao Step 3 network sweep also fixed `lambda_social=0.001` and used `boundary_downweight`.

Summary CSV:

```text
data/ciao/social_smoke_results/lambda_sweep/lambda_sweep_summary.csv
```

Plots:

![Ciao RMSE over lambda_social](../../plots/ciao/models/social_regularization/social_lambda_rmse.png)

![Ciao MAE over lambda_social](../../plots/ciao/models/social_regularization/social_lambda_mae.png)

![Ciao R2 over lambda_social](../../plots/ciao/models/social_regularization/social_lambda_r2.png)

### Step 3: Network Sweep

Step 3 tested the Ciao Step 2 winner, `boundary_downweight` with `lambda_social=0.001`, across 10 random complete network indices for each diffusion model. The sampled indices were the same reproducible index sets used for MovieLens because the available network grids have matching index coverage.

Sampled indices:

| Diffusion model | Network indices |
| --- | --- |
| `exponential` | `8`, `9`, `19`, `41`, `60`, `68`, `71`, `82`, `94`, `96` |
| `powerlaw` | `17`, `34`, `39`, `40`, `46`, `54`, `62`, `75`, `81`, `88` |
| `rayleigh` | `6`, `15`, `25`, `34`, `44`, `58`, `66`, `71`, `89`, `95` |

All 30 Ciao network-sweep jobs completed. The Step 3 aggregate below uses only rating-scale sane social-regularized rows for RMSE, MAE, and R2. RMSE deltas are computed only for rows where both the no-social control and the social-regularized run were rating-scale sane.

By-model summary over the sampled networks:

| Diffusion model | Sampled runs | Metric runs | Delta runs | Mean RMSE | RMSE std | Mean MAE | MAE std | Mean R2 | R2 std | Mean RMSE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exponential` | `10` | `10` | `10` | `0.933735696` | `0.000000179` | `0.725957393` | `0.000000077` | `0.132422715` | `0.000000333` | `-0.000000070` |
| `powerlaw` | `10` | `10` | `10` | `0.933735607` | `0.000000191` | `0.725957370` | `0.000000077` | `0.132422881` | `0.000000355` | `0.000000213` |
| `rayleigh` | `10` | `9` | `9` | `0.933735328` | `0.000000573` | `0.725957161` | `0.000000457` | `0.132423399` | `0.000001065` | `-0.000000449` |

Ciao is mostly stable at the rating-scale level, but the social-regularization effect is extremely small and occasional L-BFGS fits can still diverge. After rerunning the full Ciao network sweep with `--overwrite`, the originally suspicious `exponential` network `60` no longer produced a large error (`RMSE=0.933735433`). The fresh raw sweep instead had one unreasonable `rayleigh` network `6` fit, which was excluded from the aggregate table above.

To check whether the failures are tied to specific graph files, `exponential` network `60` and `rayleigh` network `6` were each rerun into isolated check files. Both isolated reruns were rating-scale sane, so the large values appear to be intermittent optimizer failures rather than persistent network-specific data errors.

Among the filtered Ciao Step 3 results, `rayleigh` has the best mean RMSE, MAE, R2, and mean RMSE delta. These differences are too small to rank the diffusion families confidently; the useful result is that the same low-lambda `boundary_downweight` configuration transfers from MovieLens to Ciao, provided the sweep keeps filtering or retrying rating-scale failures.

Step 3 artifacts:

```text
data/ciao/social_smoke_results/network_sweep/selected_network_indices.json
data/ciao/social_smoke_results/network_sweep/network_sweep_summary.csv
data/ciao/social_smoke_results/network_sweep/network_sweep_by_model.csv
data/ciao/social_smoke_results/network_sweep/rerun_checks/exponential_060_check.json
data/ciao/social_smoke_results/network_sweep/rerun_checks/rayleigh_006_check.json
```

Plots:

![Ciao RMSE over sampled networks](../../plots/ciao/models/social_regularization/social_network_rmse.png)

![Ciao MAE over sampled networks](../../plots/ciao/models/social_regularization/social_network_mae.png)

![Ciao R2 over sampled networks](../../plots/ciao/models/social_regularization/social_network_r2.png)

### Step 4: User-Attribute Grid

The Ciao Step 4 grid used the same constrained values as MovieLens and the same `boundary_downweight`, `lambda_social=0.001` social configuration.

Result summary:

| `lambda_reg` | `w_user` | RMSE | MAE | R2 | Baseline sane | Social sane |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1.0` | `0.01` | invalid | invalid | invalid | `true` | `false` |
| `1.0` | `0.05` | `0.972431` | `0.728258` | `0.059025` | `false` | `true` |
| `1.0` | `0.10` | `0.972062` | `0.729531` | `0.059739` | `true` | `true` |
| `3.0` | `0.01` | `0.914638` | `0.700985` | `0.167548` | `true` | `true` |
| `3.0` | `0.05` | invalid | invalid | invalid | `false` | `false` |
| `3.0` | `0.10` | `0.910226` | `0.695675` | `0.175560` | `true` | `true` |
| `10.0` | `0.01` | `0.933734` | `0.725955` | `0.132425` | `true` | `true` |
| `10.0` | `0.05` | `0.933735` | `0.725957` | `0.132423` | `true` | `true` |
| `10.0` | `0.10` | `0.933481` | `0.725756` | `0.132896` | `true` | `true` |

The best rating-scale sane Ciao Step 4 result is `lambda_reg=3.0`, `w_user=0.10` (`RMSE=0.910226`, `MAE=0.695675`, `R2=0.175560`). This again points to `lambda_reg=3.0` as the useful side-user region, with Ciao preferring a stronger side-user weight than MovieLens.

Ciao still shows intermittent L-BFGS divergence in both baseline and social-regularized fits. The invalid cells should be excluded from aggregate comparisons and either retried or handled with a stricter optimizer fallback before this grid is promoted into the full study.

Step 4 artifacts:

```text
data/ciao/social_smoke_results/user_attribute_grid/user_attribute_grid_summary.csv
data/ciao/social_smoke_results/user_attribute_grid/*.json
```

## Diagnostics

| Diagnostic | Result |
| --- | --- |
| Baseline finite metrics | `true` |
| Social-regularized finite metrics | `true` |
| Baseline rating-scale sanity | `true` |
| Social-regularized rating-scale sanity | `true` |

## Caveats

- This is a small smoke test, not a full experimental result.
- The test filters to warm users and warm items so it validates the social regularizer rather than cold-start behavior.
- User attributes are disabled in Steps 1-3 to isolate Phase 6, then enabled explicitly in Step 4.
- The Step 1 result covers one diffusion model, one network index, and four social weighting modes per dataset.
- The Step 2 result covers the same model and network index with four `lambda_social` values per mode per dataset.
- The Step 3 result covers 10 sampled network indices per diffusion model per dataset, but still uses one social mode and one `lambda_social` value per dataset.
- The Step 4 result adds user attributes back on one dataset/model/network setting per dataset and uses a constrained 3-by-3 grid over `lambda_reg` and `w_user`.
- The observed differences between social modes are far smaller than what would be needed for a model-quality claim.
- Ciao Step 3 can show intermittent L-BFGS divergence. In the overwrite rerun, the originally suspicious `exponential` network `60` was sane, while `rayleigh` network `6` diverged in the raw sweep but was sane in an isolated rerun. The by-model aggregate excludes invalid rows according to the sanity checks described above.
- Step 4 confirms that side-user attributes can improve the smoke-test metrics, but it also keeps exposing occasional L-BFGS divergence. Invalid baseline or social rows are marked by the sanity checks and should be excluded from interpretation.
- The Step 3 network plots connect sampled network indices for readability only; the x-axis is not a temporal or ordered training trajectory.
