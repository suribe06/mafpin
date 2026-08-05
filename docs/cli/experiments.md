# Experiments CLI

Batch scripts and experiment runners for the core ladder, cold-start study, and
Route B. Exact command recipes (ordered runs, log paths, gates) live in the
experiment docs; this page documents **flags and invocation**.

| Experiment docs | Commands / findings |
| --- | --- |
| [core_experiment_plan.md](../experiments/core_experiment_plan.md) | [core_experiment_commands.md](../experiments/core_experiment_commands.md) |
| [cold_start_experiment_proposal.md](../experiments/cold_start_experiment_proposal.md) | [cold_start_commands.md](../experiments/cold_start_commands.md) |
| [route_b_protocol.md](../experiments/route_b_protocol.md) | [route_b_commands.md](../experiments/route_b_commands.md) |

---

## `./scripts/run_core_experiment.sh`

Full core-experiment batch: prerequisites → hypertune → recommend → Phase 2
(`import_manifest` … `final_eval`) via `pipeline.py`.

```bash
./scripts/run_core_experiment.sh --dataset ciao
./scripts/run_core_experiment.sh --dataset movielens --from preregister
./scripts/run_core_experiment.sh --dataset ciao --dry-run
```

| Option | Default | Description |
| --- | --- | --- |
| `--dataset DATASET` | **required** | `movielens`, `ciao`, or `epinions`. |
| `--from STEP` | start of list | Resume from named step (skip earlier ones). |
| `--dry-run` | off | Print planned commands only. |
| `--n-jobs N` | `-1` | Recommend worker processes (`-1` = auto from CPU fraction). |
| `--cpu-fraction F` | `0.4` | Core fraction when `--n-jobs -1`. |
| `-h`, `--help` | — | Usage. |

**`--from` step names:**

`prerequisites`, `preregister`, `m2_hypertune`, `m3_hypertune`, `m4a_hypertune`,
`m4b_hypertune`, `m4c_hypertune`, `m4d_hypertune`, `m3_recommend`, `m4c_recommend`,
`m2_recommend`, `m4a_recommend`, `m4b_recommend`, `m4d_recommend`,
`m4c_robustness_laplacian`, `phase2_import_manifest`, `phase2_canonical_baseline`,
`phase2_network_selection`, `phase2_final_eval`.

**Environment:** `MAFPIN_PYTHON` overrides the interpreter; otherwise the script
uses the `mafpin` conda env.

Continues on failure (`set +e`). Review:

```bash
column -t -s $'\t' data/<dataset>/logs/run_summary.tsv
```

Detached runs: `tmux` / `nohup` as shown in the script `--help`.

---

## Cold-start — `python -m recommender.experiment.cold_start`

Modes: diagnostic strata on the core split, controlled rebuild with late
hold-out, zero-shot trust features, and report aggregation.

```bash
python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode diagnostic

python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode controlled \
  --split leave_last \
  --test-frac 0.2

python -m recommender.experiment.cold_start \
  --dataset ciao \
  --mode zero_shot_trust

python -m recommender.experiment.cold_start \
  --dataset movielens \
  --mode report
```

`--dataset` and `--mode` are required.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | **required** | Dataset name. |
| `--mode` | `diagnostic` \| `controlled` \| `zero_shot_trust` \| `report` | **required** | Experiment mode. |
| `--variants` | variant IDs… | mode-dependent | Override which variants to run. |
| `--seed` | int | `42` | RNG seed. |
| `--output-dir` | path | `data/<ds>/cold_start/…` | Override cold-start output root. |
| `--skip-rebuild` | flag | off | `controlled` only: skip cascade / NetInf rebuild. |
| `--bootstrap-samples` | int | `1000` | Bootstrap draws for CIs. |
| `--include-ranking` | flag | off | Also compute global ranking metrics (secondary). |
| `--test-frac` | float | `0.2` | Per-user late-holdout fraction (`controlled`). |
| `--split` | `leave_last` \| `leave_k` | `leave_last` | `controlled` split protocol. `leave_k` uses stratified caps to populate cold strata on dense data. |
| `--n-alphas` | int | `100` | NetInf α grid (rebuild path). |
| `--max-iter` | int | `5000` | NetInf edge budget fallback. |
| `--k-avg-degree` | int | `2` | Average-degree edge budget scaling. |
| `--cmf-maxiter` | int | `25` | L-BFGS iterations. |

### Default variants by mode

| Mode | Default `--variants` |
| --- | --- |
| `diagnostic`, `controlled` | `M1 M2 M3 M4c M4d` |
| `zero_shot_trust` | `M1 M2_trust M3_trust` (Ciao / Epinions) |

**Prerequisites:** core Phase 2 artefacts (`experiment_manifest.json`,
`canonical_baseline.json`, and for `diagnostic` the NetInf feature dirs). See
[cold_start_commands.md](../experiments/cold_start_commands.md).

Package README: [recommender/experiment/cold_start/README.md](../../recommender/experiment/cold_start/README.md).

---

## Route B WP2 — `python -m recommender.experiment.route_b.boundary_strata`

Boundary-user strata analysis on saved Route B predictions.

```bash
python -m recommender.experiment.route_b.boundary_strata \
  --dataset movielens \
  --variants M1 M2 M3 M4c M4d \
  --bootstrap-samples 1000 \
  --seed 42 \
  --min-n 30
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | **required** | Dataset. |
| `--bootstrap-samples` | int | `1000` | Bootstrap samples. |
| `--seed` | int | `42` | RNG seed. |
| `--min-n` | int | `30` | Minimum stratum size. |
| `--variants` | IDs… | `M1 M2 M3 M4c M4d` | Variants to analyse. |

Requires predictions from pipeline `final_eval` with `--save-predictions`.
Protocol: [route_b_protocol.md](../experiments/route_b_protocol.md).

> **Note:** Older notes may say `python -m recommender.experiment.boundary_strata`.
> The module path is `recommender.experiment.route_b.boundary_strata`.

---

## Route B WP3 — `python scripts/route_b_wp3_community_stability.py`

Community / LPH stability across α neighbours and detectors.

```bash
python scripts/route_b_wp3_community_stability.py \
  --dataset movielens \
  --neighbors 2 \
  --detectors demon aslpaw \
  --seed 42
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | dataset choices | **required** | Dataset. |
| `--neighbors` | int | `2` | α-neighbour half-width for Spearman stability. |
| `--detectors` | names… | `demon aslpaw` | Community detectors (`demon` reuses precomputed CSVs; others recompute). |
| `--seed` | int | `42` | RNG seed. |

**Output:** `data/<dataset>/route_b/community_stability.csv`.

---

## Route B via pipeline flags

WP1 / WP2 hooks on `final_eval`:

```bash
python -m pipeline --steps final_eval --dataset movielens --all-variants --beyond-accuracy
python -m pipeline --steps final_eval --dataset movielens --all-variants --save-predictions
```

See [pipeline.md](pipeline.md#core-experiment--route-b-flags).

---

## Log parser — `python scripts/parse_experiment_logs.py`

Parses core-experiment logs under `data/movielens/logs` into JSON on stdout.
**No argparse flags** (paths are hardcoded for MovieLens).

```bash
python scripts/parse_experiment_logs.py
```

---

## Variant → pipeline flag cheat sheet

| ID | Recommend / hypertune flags (beyond common CMF flags) |
| --- | --- |
| M1 | Baseline path inside `recommend` (no network side info) |
| M2 | `--no-communities` |
| M3 | default communities on |
| M4a | `--social-regularization --social-mode uniform` |
| M4b | `--social-regularization --social-mode community_jaccard` |
| M4c | `--social-regularization --social-mode boundary_downweight` |
| M4d | `--social-regularization --social-mode bridge_preserve` |
| M4c_robustness | M4c + `--social-normalization normalized_laplacian` |

Common recommend flags: `--cmf-method lbfgs --cmf-maxiter 25 --n-jobs 1 --seed 42`.
Social runs typically also set `--social-normalization mean_weight --social-search-max-ratings 0 --social-n-trials 200`.
