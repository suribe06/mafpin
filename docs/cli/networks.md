# Networks CLI

Standalone modules for cascade generation, Δ / α grids, NetInf inference,
community detection, and centrality. The [pipeline](pipeline.md) wraps the same
logic via `--steps cascade|delta|inference|communities|centrality`.

Most network CLIs default to the MovieLens dataset paths from `config.DatasetPaths`.
Centrality and communities batch modes that take `--model` / `--all` without
`--dataset` use `Datasets.DEFAULT` (`movielens`).

---

## Cascades — `python -m networks.cascades`

Generate a NetInf cascade file from ratings (train split only).

```bash
python -m networks.cascades --dataset movielens
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | `movielens` \| `ciao` \| `epinions` | `movielens` | Dataset to process. |

**Output:** `data/<dataset>/cascades.txt`.

Pipeline equivalent: `python -m pipeline --steps cascade --dataset …`.

---

## Delta — `python -m networks.delta`

Compute the median inter-event Δ and suggested α centres for each diffusion
model.

```bash
python -m networks.delta
python -m networks.delta --cascades data/ciao/cascades.txt --n-alphas 100
# also: python networks/delta.py …
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--cascades` | path | `data/movielens/cascades.txt` | Path to the cascades file. |
| `--range-r` | float | `100.0` | Multiplicative range factor for the log-spaced α grid. |
| `--n-alphas` | int | `100` | Number of α values in the grid. |

Prints α centres per model to stdout. Does not write network files.

Pipeline equivalent: `python -m pipeline --steps delta`.

---

## Inference — `python -m networks.inference`

Run NetInf over the α grid for one or all diffusion models.

```bash
python -m networks.inference --all --dataset movielens
python -m networks.inference --model exponential --dataset ciao --n-alphas 50
```

`--model` and `--all` are mutually exclusive; one is required.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `exponential` \| `powerlaw` \| `rayleigh` | — | Infer for a single model. |
| `--all` | flag | — | Infer for all three models. |
| `--dataset` | dataset choices | `movielens` | Dataset for cascades / network output dirs. |
| `--cascades` | path | dataset cascades | Override cascades file path. |
| `--n-alphas` | int | `100` | α grid size. |
| `--max-iter` | int | `5000` | Fallback edge budget *k* when average-degree scaling is off. |
| `--k-avg-degree` | float | `2` | `k = avg_degree × N`; `0` disables (use `--max-iter` only). |
| `--range-r` | float | `100.0` | Log-grid multiplicative range. |
| `--name-output` | string | `inferred-network` | Base name for per-α output files. |

**Output:** `data/<dataset>/networks/<model>/inferred-network-*.txt`.

Requires a built NetInf binary; see [installation.md](../installation.md).

Pipeline equivalent: `python -m pipeline --steps inference …`.

---

## Communities — `python -m networks.communities`

Detect overlapping communities and compute Local Pluralistic Homophily (LPH).

```bash
python -m networks.communities --all
python -m networks.communities --model exponential --algorithm demon --epsilon 0.25
python -m networks.communities --network data/movielens/networks/exponential/inferred-network-000.txt
```

Exactly one of `--model`, `--all`, or `--network` is required.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | diffusion model | — | Process all networks for that model (default dataset paths). |
| `--all` | flag | — | Process all models. |
| `--network` | file path | — | Process a single network file. |
| `--algorithm` | `demon` \| `aslpaw` | `demon` | Community detection algorithm. |
| `--epsilon` | float | `0.25` | Demon merging threshold. |
| `--min-community` | int | `3` | Minimum community size (Demon). |
| `--symmetrization` | `union` \| `intersection` | `union` | Directed → undirected conversion before detection / LPH. `union`: edge if either direction; `intersection`: both directions required. |
| `--boundary-percentile` | float `0–100` | `20.0` | Percentile for binary `is_boundary`: nodes with ĥᵥ at or below this percentile are boundary-spanners. |

**Output:** community / LPH CSVs under `data/<dataset>/communities/<model>/`.

Background: [lph.md](../lph.md).

Pipeline equivalent: `python -m pipeline --steps communities`.

---

## Centrality — `python -m networks.centrality`

Compute SNAP centrality metrics on inferred networks.

```bash
python -m networks.centrality --all
python -m networks.centrality --model rayleigh
python -m networks.centrality --network path/to/inferred-network-012.txt
```

Exactly one of `--model`, `--all`, or `--network` is required. There is no
`--dataset` flag; batch modes use default dataset paths.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | diffusion model | — | All networks for one model. |
| `--all` | flag | — | All models. |
| `--network` | file path | — | Single network file. |

Metrics: degree, betweenness, closeness, eigenvector, PageRank, clustering,
eccentricity. See [centrality_metrics.md](../centrality_metrics.md).

**Output:** `data/<dataset>/centrality_metrics/<model>/`.

Pipeline equivalent: `python -m pipeline --steps centrality`.

---

## Suggested standalone order

```bash
python -m networks.cascades --dataset movielens
python -m networks.delta --cascades data/movielens/cascades.txt
python -m networks.inference --all --dataset movielens
python -m networks.communities --all
python -m networks.centrality --all
```

Prefer the pipeline when you also need logging, `--k-avg-degree` consistency
with later recommend steps, or a single command for the full chain.
