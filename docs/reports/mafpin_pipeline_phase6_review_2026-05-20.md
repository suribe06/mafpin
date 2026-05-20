# MAFPIN Pipeline Review Report

**Date**: May 20, 2026  
**Reviewed by**: AI Code Review Agent  
**Pipeline version**: feat/research-phase-6 branch; no explicit semantic version found

---

## Executive Summary

The Phase 6 social-regularized CMF extension is conceptually strong and broadly aligned with graph-regularized matrix factorization. The implemented C objective adds a weighted social edge penalty over user latent factors and uses L-BFGS, which is the correct optimizer family for a coupled-user objective. The social weighting modes also give the project a credible methodological story: uniform smoothing as a baseline, community agreement as homophily-aware smoothing, and boundary-aware weights as a way to reduce oversmoothing around users at community interfaces.

The pipeline is not yet ready for final experiments because two critical validity issues can contaminate reported results. First, cascade generation does not use the same configured global split as recommender evaluation, so inferred networks can be built from interactions that should belong to the recommendation test period. Second, social hyperparameter search reloads and splits the full dataset independently, allowing the global held-out ratings to affect model selection. These should be fixed and all generated artifacts should be regenerated before interpreting Phase 6 metrics.

After those leakage issues are fixed, the pipeline becomes viable for publishable research, but the final study should include stronger ablations, per-mode social regularization comparisons, multi-seed stability checks, and clearer reporting of SHAP surrogate quality. The recommended priority is to harden the split/artifact contract first, then enforce optimizer consistency, then improve social search and reporting.

---

## Issue Registry

| # | Location | Severity | Category | Summary |
| --- | ---------- | ---------- | ---------- | --------- |
| 1 | `pipeline.py::_run_cascade` | Critical | Leakage | Cascade generation uses a random split instead of the configured global split used by recommendation. |
| 2 | `recommender/enhanced/social_search.py::_prepare_search_data` | Critical | Leakage | Social hyperparameter search reloads the full dataset and creates an independent split, exposing global test ratings to tuning. |
| 3 | Generated artifacts under `data/<dataset>/` | Critical | Leakage / Reproducibility | Precomputed cascades, networks, centrality, communities, LPH, and SHAP artifacts can be stale relative to the current split protocol. |
| 4 | `pipeline.py` CLI and social recommend/hypertune/shap steps | Major | Correctness | `--social-regularization --cmf-method als` is accepted even though social CMF requires L-BFGS. |
| 5 | `recommender/enhanced/network_eval.py::evaluate_social_cmf_with_user_attributes` | Major | Correctness | The paired baseline in the social path uses `Defaults.CMF_METHOD` instead of the method chosen by the pipeline. |
| 6 | `recommender/enhanced/network_eval.py::run_network_evaluation` | Major | Robustness | Network sampling is based on `range(len(csvs))`, which can select missing indices when artifact generation skipped networks. |
| 7 | `recommender/enhanced/network_eval.py::_save_rmses` | Major | Robustness | Results are written back into shared `inferred_edges_*.csv` files without run-mode separation, risking stale enhanced/social mixing. |
| 8 | JSON, CSV, and NPY writers across pipeline | Major | Robustness | Artifacts are written directly rather than atomically; interrupted runs can leave partial or misleading files. |
| 9 | `recommender/enhanced/social_search.py::_trial_params` | Major | Hyperparameter Search | `beta` and `gamma` are sampled even for modes where they have no effect. |
| 10 | `recommender/enhanced/social_search.py::search_social_regularized_params` | Major | Hyperparameter Search | Social search is run on one representative network, which may not generalize across diffusion models or alpha values. |
| 11 | Social regularization objective and edge construction | Major | Methodology | `lambda_social` is not explicitly scaled by edge count, total edge weight, degree distribution, or rating sparsity. |
| 12 | `recommender/enhanced/model.py` vs `network_eval.py` | Major | Evaluation | Warm-test filtering is stricter in the social path than the non-social enhanced path. |
| 13 | `pipeline.py` alpha plots | Major | Evaluation / Reporting | Alpha-RMSE plots receive the global baseline but label it as the paired baseline. |
| 14 | `analysis/shap_analysis.py::run_shap_analysis` | Major | SHAP | Skipped SHAP networks due to low surrogate R2 are printed but not saved as structured audit metadata. |
| 15 | `pipeline.py::_run_shap` and `analysis/shap_analysis.py::_train_enhanced_cmf` | Major | SHAP / Reporting | SHAP loads social best params correctly, but MLflow logs CLI fallback social params rather than the loaded best social params. |
| 16 | `visualization/shap_plots.py` | Major | SHAP / Reporting | SHAP titles and filenames do not distinguish enhanced CMF from social-regularized CMF. |
| 17 | `recommender/baseline.py::search_baseline_params` and `recommender/enhanced/search.py::search_enhanced_params` | Major | Reproducibility | Baseline and enhanced Optuna searches are unseeded while social search is seeded. |
| 18 | `recommender/enhanced/social_search.py` | Minor | Observability | Social Optuna trials do not log trial metrics or best params to MLflow. |
| 19 | `pipeline.py` step ordering and docs | Minor | Correctness / Documentation | The docstring says centrality requires communities first; the actual all-step order runs communities before centrality, but user examples can still run centrality alone. |
| 20 | `analysis/shap_analysis.py` | Minor | SHAP | SHAP uses global test interactions to define per-user prediction targets; acceptable for explanation, but should be documented as post-hoc only. |
| 21 | Social weighting modes | Suggestion | Methodology | `social_mode` is searched as one categorical variable, but final claims will be cleaner if modes are evaluated as controlled conditions. |
| 22 | Final experiments | Suggestion | Evaluation | Final results should use all valid networks or a pre-registered stratified sample, not ad hoc sampled networks. |

---

## Detailed Findings

### Leakage

#### Issue 1: Cascade split does not match the configured global split

**Location**: `pipeline.py::_run_cascade`; cascade step  
**Severity**: Critical  
**Description**: The cascade step directly calls `train_test_split` with a random split. The recommendation path uses `load_and_split_dataset`, which respects `config.Split.STRATEGY` and is currently documented as the global split contract. If `Split.STRATEGY` is temporal, the cascades and inferred networks are not built from the same training period as the recommender. This can put future/test interactions into the network inference stage and invalidate downstream centrality, community, LPH, social-edge, and SHAP artifacts.  
**Fix**: Replace the direct random split in `_run_cascade` with `load_and_split_dataset(dataset=args.dataset)` and pass its `train_df` to `generate_cascades_from_df`. Ensure the full user ID list still comes from the full encoded dataset so compact NetInf IDs remain aligned with recommender user IDs.

#### Issue 2: Social hyperparameter search uses the full dataset independently

**Location**: `recommender/enhanced/social_search.py::_prepare_search_data`; `pipeline.py::_run_recommend`; `pipeline.py::_run_hypertune`  
**Severity**: Critical  
**Description**: The social search reloads the dataset internally, filters by feature users, optionally samples ratings, and then creates a new random train/test split. In the pipeline, social search is called with only the dataset name, not the already constructed global training split. This means the global held-out test data can influence social hyperparameter selection. This is the highest-risk Phase 6 issue because it can make social regularization appear better than it is.  
**Fix**: Refactor `search_social_regularized_params` to accept a `data` or `train_df` argument and use only the pipeline's global training split. Keep its internal validation split as a validation split of global train only. Update `recommend` and `hypertune` to pass the existing `train_df`. Record `source_split="global_train"` in the search JSON.

#### Issue 3: Precomputed artifacts may encode outdated split behavior

**Location**: Generated artifacts under `data/<dataset>/cascades.txt`, `inferred_networks/`, `centrality_metrics/`, `communities/`, `shap_matrices/`, and search-result JSON files  
**Severity**: Critical  
**Description**: The pipeline loads generated artifacts from dataset-specific paths, which is good, but it has no manifest tying those artifacts to a split strategy, random seed, temporal cutoff, dataset file hash, code version, or social search configuration. After changing split logic, old cascades and networks can still be reused silently. This creates hidden leakage and irreproducibility.  
**Fix**: Add a manifest file under `data/<dataset>/artifact_manifest.json` containing dataset, source file path/hash, split strategy, split seed or temporal cutoff, cascade generation timestamp, network inference parameters, and git branch/commit when available. Each downstream step should validate the manifest before loading artifacts. Regenerate all artifacts after fixing Issues 1 and 2.

### Correctness and Implementation Bugs

#### Issue 4: Social regularization accepts ALS in the CLI

**Location**: `pipeline.py` argument parsing and social recommend/hypertune/shap steps  
**Severity**: Major  
**Description**: The CLI allows `--cmf-method als` with `--social-regularization`. Social CMF itself forces `method="lbfgs"`, but baseline/enhanced paths can still use the CLI method. That breaks the intended optimizer-consistency claim and can make runs misleading.  
**Fix**: Add validation after parsing: if `args.social_regularization and args.cmf_method != "lbfgs"`, either raise a clear error or force `args.cmf_method = "lbfgs"` and log the override. Prefer an error for final experiments.

#### Issue 5: Social paired baseline ignores the pipeline method argument

**Location**: `recommender/enhanced/network_eval.py::evaluate_social_cmf_with_user_attributes`  
**Severity**: Major  
**Description**: The paired baseline inside social network evaluation calls `train_model(..., method=Defaults.CMF_METHOD)`. This happens to be L-BFGS today, but it is not explicitly tied to the current pipeline run. If defaults or CLI choices change, the paired social comparison can become inconsistent.  
**Fix**: Add a `method` argument to `evaluate_social_cmf_with_user_attributes` and pass it through from `evaluate_single_network`. Then use that method for the paired baseline. Also enforce L-BFGS when social is enabled.

#### Issue 6: Network index sampling can select nonexistent artifact indices

**Location**: `recommender/enhanced/network_eval.py::run_network_evaluation`  
**Severity**: Major  
**Description**: The code finds centrality CSV files and then uses `indices = list(range(len(csvs)))`. If centrality generation skipped network `003` but produced `004`, the sampled index can point to the wrong or missing artifact. Social runs are even more sensitive because they also require network and community files.  
**Fix**: Parse network indices from file names and sample actual available indices. For social evaluation, intersect valid indices across inferred network, centrality, and community artifacts. Reuse the stricter availability logic already present in `social_network_sweep.py`.

#### Issue 7: Enhanced and social results share the same inferred-edge summary columns

**Location**: `recommender/enhanced/network_eval.py::_save_rmses`; `data/<dataset>/inferred_networks/<model>/inferred_edges_*.csv`  
**Severity**: Major  
**Description**: Network evaluation writes `rmse_mean`, `baseline_rmse_mean`, `improvement_pct`, and ranking metrics back into the same inferred-edge summary file. A social run can overwrite values from an enhanced run, and plots may later read mixed or stale values.  
**Fix**: Add a `result_prefix` or `run_mode` parameter and write separate columns such as `enhanced_rmse_mean`, `social_rmse_mean`, `enhanced_ndcg_at_k`, `social_ndcg_at_k`. Alternatively, write evaluation outputs to separate `enhanced_network_eval.csv` and `social_network_eval.csv` files.

#### Issue 8: Artifact writes are not atomic

**Location**: `save_social_search_results`, `save_enhanced_search_results`, `save_search_results`, `save_shap_results`, `np.save` for SHAP matrices, CSV writers in network/evaluation steps  
**Severity**: Major  
**Description**: Most outputs are written directly to their final path. If a run is interrupted, later steps may read a partial JSON, incomplete CSV, or a matrix saved for only some networks. Long social searches and SHAP runs are especially exposed.  
**Fix**: Write to a temporary file in the same directory, flush/close it, then atomically replace the destination with `Path.replace`. For SHAP matrices, include the mode in the filename and write a completion manifest after all matrices are saved.

#### Issue 19: Centrality/community step ordering is easy to misuse

**Location**: `pipeline.py` docstring, `STEPS`, and usage examples  
**Severity**: Minor  
**Description**: The pipeline docstring says centrality requires communities first so `pagerank_lph` can be included. The all-step order runs communities before centrality, but examples such as `--steps inference centrality` can produce centrality files without `pagerank_lph`. That is not fatal, but it changes feature columns across runs.  
**Fix**: Add a warning in `_run_centrality` when matching community files are missing. Consider making `centrality` depend on `communities` unless the user passes an explicit `--allow-missing-lph` flag.

### Evaluation Validity

#### Issue 10: Social search is tuned on one representative network

**Location**: `pipeline.py::_run_recommend`; `pipeline.py::_run_hypertune`; `recommender/enhanced/social_search.py::search_social_regularized_params`  
**Severity**: Major  
**Description**: The social search tunes eight parameters on one network index, normally the first available network for the first selected diffusion model. Social-edge quality, graph density, community overlap, and edge-retention behavior can vary across alpha values and diffusion models. A single-network search can overfit to that network's topology.  
**Fix**: For final experiments, tune on a small stratified validation set of networks spanning sparse, medium, and dense alpha regions across diffusion models. At minimum, validate the selected parameters on a held-out set of networks before final reporting.

#### Issue 12: Warm-test filtering differs between social and non-social paths

**Location**: `recommender/enhanced/model.py::evaluate_cmf_with_user_attributes`; `recommender/enhanced/network_eval.py::evaluate_social_cmf_with_user_attributes`  
**Severity**: Major  
**Description**: The social evaluation explicitly filters each fold's test set to users and items seen in train. The non-social enhanced evaluator does not apply the same warm-test filtering. This can make RMSE and ranking metrics not directly comparable across enhanced and social paths.  
**Fix**: Centralize fold creation and warm-test filtering in a shared utility. Use the same fold splitter for baseline, enhanced, social, and SHAP training/evaluation.

#### Issue 13: Alpha plots label global baseline as paired baseline

**Location**: `pipeline.py` plotting section; `visualization/model_plots/alpha.py`  
**Severity**: Major  
**Description**: `pipeline.py` passes `baseline_metrics["rmse"]` from the global final baseline into `plot_alpha_rmse_analysis`, where the label says paired baseline. This conflates two different comparison baselines: global held-out tuned baseline and per-network paired CV baseline.  
**Fix**: Pass the mean paired baseline from network evaluation to paired-baseline plots and optionally pass global baseline separately via `global_baseline_rmse`. Update plot labels and saved filenames to distinguish global vs paired comparisons.

#### Issue 22: Sampled networks are acceptable for development but weak for final claims

**Location**: `pipeline.py --sample-networks`; final experiment protocol  
**Severity**: Suggestion  
**Description**: Sampling five networks per model is practical during development but insufficient for final claims unless the sample is pre-registered and stratified. Alpha-level results can be sensitive to graph density and diffusion model.  
**Fix**: For final experiments, use all valid networks. If runtime makes that infeasible, use a pre-registered stratified sample across alpha quantiles and diffusion models, and report confidence intervals.

### Methodological Correctness

#### Issue 11: Social regularization scale is not explicitly normalized to graph/rating size

**Location**: `recommender/enhanced/social_regularization.py::build_social_edges`; C social objective in `cmfrec-master/src/collective.c`  
**Severity**: Major  
**Description**: Edge weights are normalized to mean one, which helps compare weighting modes. However, the total social penalty still scales with the number of retained edges, degree distribution, and rating sparsity. A `lambda_social` value can mean different things across networks, datasets, and modes.  
**Fix**: Add normalization options: divide the social penalty by `sum_w`, by `n_edges`, or by a rating-scale factor such as `n_ratings`. Add a normalized Laplacian option using `D^-1/2 W D^-1/2`. Report which normalization is used in all results.

#### Issue 21: Social modes are searched jointly but should also be controlled conditions

**Location**: `recommender/enhanced/social_search.py::_trial_params`; final experiment design  
**Severity**: Suggestion  
**Description**: Searching `social_mode` as one categorical variable can find a good setting, but it weakens methodological interpretation. If boundary-aware weighting wins, reviewers will want to know whether that is due to boundary logic, community overlap, or general smoothing.  
**Fix**: Run per-mode searches or a two-stage search: tune each mode separately with conditional parameters, then compare each mode under its best validation configuration. Treat `uniform` as the graph Laplacian baseline.

### Hyperparameter Search

#### Issue 9: `beta` and `gamma` are sampled unnecessarily for some social modes

**Location**: `recommender/enhanced/social_search.py::_trial_params`  
**Severity**: Major  
**Description**: `uniform` ignores both `beta` and `gamma`; `community_jaccard` ignores both; `boundary_downweight` ignores `gamma`. Sampling inactive dimensions wastes trials and makes Optuna's search statistics harder to interpret. In an eight-parameter search with 200 trials, this matters.  
**Fix**: Make the search conditional. Sample `social_mode` first, then sample only parameters used by that mode. Save inactive parameters as `null` or omit them from the trial row.

#### Issue 17: Baseline and enhanced Optuna searches are unseeded

**Location**: `recommender/baseline.py::search_baseline_params`; `recommender/enhanced/search.py::search_enhanced_params`  
**Severity**: Major  
**Description**: Social search uses `TPESampler(seed=random_state)`, but baseline and enhanced searches call `optuna.create_study(direction="minimize")` without a seeded sampler. This prevents exact reruns and weakens comparisons across baseline, enhanced, and social tuning.  
**Fix**: Add a `random_state` argument to baseline and enhanced search functions and instantiate `optuna.samplers.TPESampler(seed=random_state)`.

#### Issue 18: Social search does not log trial metrics to MLflow

**Location**: `recommender/enhanced/social_search.py::search_social_regularized_params`  
**Severity**: Minor  
**Description**: Baseline and enhanced searches log Optuna metrics to MLflow when an active run exists. Social search saves JSON and prints trial summaries but does not log trial RMSE, best social params, or social-edge diagnostics to MLflow. This makes long Phase 6 runs harder to debug.  
**Fix**: Add MLflow logging inside the social objective and after search completion: `social_trial_rmse`, best `k`, `lambda_reg`, `w_main`, `w_user`, `lambda_social`, `social_mode`, `beta`, `gamma`, `n_edges`, and weight diagnostics.

### SHAP Interpretation and Reporting

#### Issue 14: SHAP skipped-network audit is not persisted

**Location**: `analysis/shap_analysis.py::compute_shap_for_network`; `analysis/shap_analysis.py::run_shap_analysis`  
**Severity**: Major  
**Description**: Networks with insufficient data or low surrogate R2 are skipped. The skip is printed to the console, but final `shap_results.json` only records successfully processed network indices. This makes the SHAP summary look cleaner than the actual run and prevents later audit of surrogate reliability.  
**Fix**: Return a structured result object with `status`, `reason`, `surrogate_r2`, `n_users`, and `network_index`. Save `skipped_networks`, `n_attempted`, `n_valid`, and surrogate R2 summaries in `shap_results.json`.

#### Issue 15: SHAP MLflow social params can differ from loaded best params

**Location**: `pipeline.py::_run_shap`; `analysis/shap_analysis.py::_train_enhanced_cmf`  
**Severity**: Major  
**Description**: When social regularization is enabled, SHAP passes `params_path=dp.SOCIAL_RESULTS` and the training helper correctly uses `params.get("social_mode")`, `params.get("lambda_social")`, `params.get("beta")`, and `params.get("gamma")`. However, the `shap` MLflow run logs CLI fallback values before loading the search result. The tracking metadata can therefore describe a different social model than the one actually explained.  
**Fix**: Load best params before logging SHAP params, or have `run_shap_analysis` return the resolved social params and log them after the call.

#### Issue 16: SHAP plots do not distinguish enhanced vs social CMF

**Location**: `visualization/shap_plots.py`  
**Severity**: Major  
**Description**: Plot titles say enhanced CMF even when SHAP was run for social-regularized CMF. Output filenames are also shared, so social SHAP can overwrite enhanced SHAP plots. This weakens interpretation and can accidentally mix figures in the article.  
**Fix**: Add `model_variant` metadata to `shap_results.json` and plot titles. Save social plots as `social_shap_importance_comparison.png` and `social_shap_beeswarm_<model>.png`, or include mode and lambda in filenames.

#### Issue 20: SHAP target uses test predictions and should be documented as post-hoc

**Location**: `analysis/shap_analysis.py::compute_shap_for_network`  
**Severity**: Minor  
**Description**: SHAP builds per-user targets from model predictions on the global test interactions. This is acceptable for post-hoc interpretation if it does not feed back into model selection, but it should be transparent because the surrogate target is not a deployment-time training signal.  
**Fix**: Document this in the report and JSON metadata. Ensure SHAP results are never used to choose hyperparameters or select networks for final model performance claims.

---

## Prioritized Action Plan

1. Fix the split contract first. Update cascade generation to use the same global split loader as recommendation, and update social search to tune only on global training data. These two fixes remove the most serious leakage risks.

2. Invalidate and regenerate artifacts. After the split fixes, delete or archive generated cascades, inferred networks, centrality files, communities, LPH files, search results, SHAP matrices, and plots. Add a manifest before rerunning expensive steps.

3. Enforce L-BFGS for social regularization. Add CLI validation so social runs cannot accidentally mix ALS baseline/enhanced fits with L-BFGS social fits.

4. Make evaluation outputs variant-safe. Separate enhanced and social result columns or files, and update alpha plots to distinguish global and paired baselines.

5. Fix valid-index sampling. Sample parsed network indices and require complete network, centrality, and community artifacts for social evaluation.

6. Make social search conditional and reproducible. Seed baseline/enhanced Optuna samplers, conditionally sample `beta` and `gamma`, and log social trial metrics to MLflow.

7. Improve social formulation robustness. Add edge-count or total-weight normalization for `lambda_social`, plus an optional normalized Laplacian formulation.

8. Harden SHAP reporting. Persist skipped-network reasons, surrogate R2 values, model variant metadata, and social params used by the actual explained model.

9. Run final ablations only after the above are complete. Use all valid networks or a pre-registered stratified sample, and repeat key social configurations over multiple seeds.

---

## Recommended Experimental Plan

1. Plain CMF baseline with separately tuned `k` and `lambda_reg`.

2. Enhanced CMF with centrality-only side-user features.

3. Enhanced CMF with centrality plus LPH features.

4. Enhanced CMF with centrality plus community membership features.

5. Enhanced CMF with all side-user features.

6. Social-only CMF with uniform graph regularization and no side-user matrix.

7. Social-only CMF with `community_jaccard`, `boundary_downweight`, and `bridge_preserve` evaluated as separate conditions.

8. Enhanced plus social CMF for each social weighting mode.

9. Diffusion model comparison across exponential, powerlaw, and rayleigh networks.

10. Stability runs over at least three random seeds for social search and final evaluation.

Final reporting should include RMSE, MAE, ranking metrics, paired baseline deltas, confidence intervals across networks, and surrogate-quality summaries for SHAP.

---

## Concrete Improvements to Social Regularization

- Add `social_normalization` choices: `none`, `mean_weight`, `sum_weight`, `n_edges`, and `normalized_laplacian`.
- Treat `uniform` as the baseline graph Laplacian condition.
- Tune each social mode separately, then compare modes under a locked validation protocol.
- Report edge diagnostics per network: usable edges, retained edge fraction, mean/min/max weights, weighted degree distribution, and number of connected components after weighting.
- Add an ablation with social regularization only and no side-user attributes to separate the effect of direct graph smoothing from centrality/community/LPH features.
- Consider directed social regularization as a future extension, but keep the current undirected Laplacian if the methodological framing is homophily/smoothing rather than directional influence.

---

## Concrete Improvements to Pipeline Implementation

- Add a shared split utility used by cascade, baseline, enhanced, social, network evaluation, and SHAP.
- Add artifact manifests and validate them before loading generated files.
- Add atomic write helpers for JSON, CSV, and NPY artifacts.
- Add `run_mode` or `model_variant` metadata to search results, network evaluation outputs, plots, SHAP results, and MLflow params.
- Seed all Optuna studies and network sampling paths.
- Store complete SHAP audit metadata, including skipped networks and surrogate R2.
- Create separate plot filenames for enhanced and social variants.
- Make social search resumable with an Optuna SQLite storage path for long 200+ trial runs.

---

## Research Contribution Notes

The strongest contribution is not simply adding another regularizer; it is the boundary-aware interpretation of when social smoothing should be reduced or preserved. To make that contribution convincing, the final paper should show that boundary-aware weighting behaves differently from uniform smoothing and plain community-overlap smoothing. A clean ablation table will matter more than a single best RMSE number.

Community features should not be removed only because SHAP assigns them low direct contribution in the side-user matrix. In the Phase 6 framing, communities may be more valuable as edge-weight structure than as direct CMF side attributes. Treat these as separate channels: direct user descriptors, social-edge weights, and combined models.

The final methodology should explicitly separate four sources of signal: centrality, community membership, LPH/boundary scores, and social Laplacian regularization. The best experimental story is a progression from plain CMF to feature-enhanced CMF to social-regularized CMF, with controlled social modes and multiple diffusion networks. That structure will make the Phase 6 novelty much easier to defend.

---

## Must-Fix Before Final Experiments

1. Align cascade generation with the configured global split.
2. Restrict social hyperparameter search to global training data only.
3. Regenerate all artifacts after fixing the split contract.
4. Enforce L-BFGS for every social-regularized pipeline run.
5. Prevent stale enhanced/social result mixing in network summaries and plots.
6. Seed all hyperparameter searches and sampled-network selection.

## Should-Fix Before Writing Results

1. Conditional social search by mode.
2. Edge-count or total-weight normalization for social regularization.
3. Structured SHAP skipped-network reporting.
4. Variant-aware SHAP plot titles and filenames.
5. Atomic artifact writes and artifact manifests.
6. Consistent warm-test filtering across baseline, enhanced, social, ranking, and SHAP evaluation.

## Nice-to-Have

1. Normalized Laplacian social penalty.
2. Multi-seed stability table for social search.
3. Per-mode social regularization reports.
4. Explicit trust-graph comparisons for Ciao and Epinions.
5. Resumable Optuna studies for long social searches.
