# MAFPIN Pipeline Audit Report

**Auditor:** GitHub Copilot (Claude Sonnet 4.6)  
**Date:** 2026-03-31  
**Scope:** Full pipeline — cascade generation → alpha grid → NetInf inference →
centrality → communities/LPH → CMF baseline + enhanced

---

## Executive Summary

Three **critical** bugs and three **major** issues were found.  Every one of
the critical bugs independently invalidates the side-information passed to the
enhanced CMF model, which fully explains why results do not improve over the
baseline.  Fix the three critical issues first; then re-run the full pipeline
from scratch.

---

## Critical Issues

---

### C-1 — Cascade sorted in descending temporal order (seed node reversed)

| | |
|---|---|
| **File** | `networks/cascades.py` |
| **Function** | `generate_cascades` |
| **Severity** | CRITICAL |

**Description**

```python
# line 96
records.sort(key=lambda x: x[1], reverse=True)   # ← reverse=True
```

Every cascade is written with the **most-recently infected** user listed first
and the **earliest infected** (seed) user listed last.  NetInf's diffusion
model defines a valid transmission `(u → v)` as one where `t_u < t_v`; it
expects the seed to appear first in the cascade so that positional order
confirms temporal order.  With `reverse=True` the roles are inverted: the
"source" of each cascade in the file is the last infected node, and NetInf
will attempt to fit edges that point **backwards in time**.  Every inferred
network in `data/inferred_networks/` was built from incorrectly ordered
cascades.

**Suggested fix**

```python
records.sort(key=lambda x: x[1])   # ascending (earliest first)
```

Regenerate `cascades.txt` and re-run all downstream steps.

---

### C-2 — User ID off-by-one: network compact IDs (1-based) vs. rating matrix
IDs (0-based)

| | |
|---|---|
| **Files** | `networks/network_io.py`, `recommender/enhanced.py` |
| **Functions** | `_build_mapper`, `evaluate_cmf_with_user_attributes` |
| **Severity** | CRITICAL |

**Description**

Two independent ID-encoding decisions create a systematic off-by-one that
silently assigns every user the **wrong** network features.

Step-by-step trace (671 users, original IDs 1–671; 9 066 items):

| Stage | Mapping | Result for original user `k` |
|---|---|---|
| `recommender/data.py` `LabelEncoder` | `original k → rating index k-1` | Rating user 0 = original user 1, rating user 1 = original user 2, … |
| `networks/cascades.py` `user_mapper` | `original k → 9065 + k` (0-based offset past items) | Cascade node 9066 = original user 1, 9067 = original 2, … |
| `networks/network_io.py` `_build_mapper` | compact 1-based sort | Compact node 1 = original user 1, compact 2 = original user 2, … |
| `centrality CSV` | stores compact ID | `UserId` 1 = original user 1, `UserId` 2 = original user 2, … |

The correct alignment is: **rating user `i` ↔ compact network node `i+1`**.

In `evaluate_cmf_with_user_attributes` the lookup is:

```python
valid_users = set(user_attributes.index)   # {1, 2, …, 671}
filtered = data[data["UserId"].isin(valid_users)]  # keeps rating users {1…670}; drops user 0
u_matrix = user_attributes.loc[
    [u for u in train_users if u in user_attributes.index]  # rating user i → network features for node i
]
```

Result:
- Rating user 0 (original user 1) is **silently dropped** from evaluation.
- Rating user `i` (original user `i+1`) receives the features of network node
  `i` (original user `i`) — a one-position shift that persists across all 670
  remaining users.

Every user's side information belongs to a different person.  The enhanced
CMF is trained on hallucinated side information.

**Suggested fix**

The cleanest fix is to make the cascade user mapping start at 0 (eliminating
the items-offset) and make `_build_mapper` return 0-based IDs, so the network
space and the LabelEncoder space are identical:

```python
# networks/network_io.py
def _build_mapper(nodes: list[int]) -> dict[int, int]:
    """Return a mapping from original IDs to compact 0-based integers."""
    return {old: new for new, old in enumerate(sorted(set(nodes)), start=0)}
```

Then update `G.AddNode` / `G.add_nodes_from` calls (SNAP requires node IDs
≥ 0; `TUNGraph` accepts 0).  Alternatively, keep 1-based SNAP IDs and apply a
`+1` shift when looking up features in the enhanced recommender:

```python
# recommender/enhanced.py — inside evaluate_cmf_with_user_attributes
# shift: rating user i  →  network compact node i+1
valid_users_rating = {u - 1 for u in user_attributes.index}   # convert to 0-based
filtered = data[data["UserId"].isin(valid_users_rating)]
...
u_matrix = user_attributes.rename(index=lambda x: x - 1).loc[
    [u for u in train_users if (u + 1) in user_attributes.index]
].reset_index()
```

Re-run centrality and community steps after fixing `_build_mapper` so the
stored CSV files use the corrected ID space.

---

### C-3 — Cascade generation uses the full dataset; test interactions contaminate
the inferred network

| | |
|---|---|
| **File** | `networks/cascades.py` |
| **Function** | `generate_cascades` |
| **Severity** | CRITICAL |

**Description**

```python
interactions = pd.read_csv(data_file, usecols=range(4))  # entire CSV — no split
```

The cascade file is built from **all** user–item interactions before any
train/test split.  Every downstream artefact (inferred networks, centrality,
LPH) therefore contains information derived from test-set ratings.  When the
enhanced CMF is later evaluated on held-out test rows, those ratings have
already influenced the network topology that supplies the user side
information.  This is a direct form of **data leakage**: the model indirectly
"sees" test-set behaviour during training.

In addition, `generate_cascades` is called once (during the `cascade` step)
entirely independently from the `recommend` step where the train/test split
is created, so it is structurally impossible for the two steps to share the
same split.

**Suggested fix**

Move cascade generation inside the recommender evaluation loop so that it
operates only on training interactions:

```python
# In recommender/enhanced.py (or pipeline.py)
train_df, test_df = split_data_single(data, test_size=0.2, random_state=42)
generate_cascades_from_df(train_df)          # new helper that accepts a DataFrame
# …then run inference, centrality, LPH on the train-only cascade
```

Add a `generate_cascades_from_df(df: pd.DataFrame) -> bool` variant to
`networks/cascades.py` that accepts an already-split DataFrame instead of
reading from disk.

---

## Major Issues

---

### M-1 — Median delta computed from all pairwise time differences, not
consecutive ones

| | |
|---|---|
| **File** | `networks/delta.py` |
| **Function** | `compute_median_delta` |
| **Severity** | MAJOR |

**Description**

```python
for i in range(len(timestamps)):
    for j in range(i + 1, len(timestamps)):
        diff = abs(timestamps[j] - timestamps[i])   # all pairs
        if diff > 0:
            deltas.append(diff)
```

The code accumulates `O(n²)` pairwise absolute time differences per cascade.
Multi-hop intervals (e.g., the gap between the 1st and 10th infected node) are
included alongside consecutive gaps.  The median of all pairwise differences
is substantially larger than the median of consecutive (adjacent) differences,
because long multi-hop spans dominate the sample.

Consequence: `delta_seconds` is overestimated → `alpha_center = ln(2)/delta`
is biased downward → the log-alpha grid is centred on a value that is too
small → NetInf is swept through predominantly slow-decay, high-density
regimes, producing overly dense networks.

**Suggested fix**

Collect only consecutive (adjacent) inter-event differences within each
cascade (assuming ascending sort after C-1 fix):

```python
timestamps.sort()  # ensure ascending after C-1 fix
for i in range(len(timestamps) - 1):
    diff = timestamps[i + 1] - timestamps[i]
    if diff > 0:
        deltas.append(diff)
```

---

### M-2 — Feature scaler fitted on all users before the train/test split

| | |
|---|---|
| **File** | `recommender/enhanced.py` |
| **Function** | `load_network_features` |
| **Severity** | MAJOR |

**Description**

```python
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.values),   # fit on ALL users, including test users
    index=df.index,
    columns=df.columns,
)
return df_scaled
```

`load_network_features` is called once per network, outside the
`split_data_single` loop.  The scaler's mean and variance are computed over
**all** users (including those that will appear only in the test set).  This
is a subtle but standard form of leakage: test-user feature statistics
influence the normalisation applied to training data.

**Suggested fix**

Move normalisation inside the per-split loop and fit the scaler only on
training users:

```python
# inside evaluate_cmf_with_user_attributes, after train_df / test_df split
train_users = set(train_df["UserId"].unique())
train_feats = raw_features.loc[raw_features.index.isin(train_users)]
scaler = StandardScaler().fit(train_feats.values)
scaled_all = pd.DataFrame(
    scaler.transform(raw_features.values),
    index=raw_features.index, columns=raw_features.columns
)
# then use scaled_all for both u_matrix (train) and any test-user lookup
```

Pass `raw_features` (unscaled) into `evaluate_cmf_with_user_attributes` and
perform the fit-transform inside the loop.

---

### M-3 — Enhanced RMSE compared against a baseline measured on a different
(larger) user set

| | |
|---|---|
| **File** | `recommender/enhanced.py` |
| **Function** | `run_network_evaluation` |
| **Severity** | MAJOR |

**Description**

```python
search_result = search_best_params(data, n_iter=50, n_splits=3)   # all 671 users
baseline_rmse = float(min(r["rmse"] for r in search_result["all_results"]))

# later, inside evaluate_cmf_with_user_attributes:
filtered = data[data["UserId"].isin(list(valid_users))]   # only 670 users
```

The baseline RMSE is computed on the full 671-user dataset.  The enhanced
RMSE is computed on a 670-user subset (user 0 is dropped due to the C-2
mismatch; and any user absent from the network is excluded).  These two
quantities measure performance on populations that are not comparable.  If the
dropped user has above-average RMSE, the enhanced model will appear better
purely because it was evaluated on an easier subset.

**Suggested fix**

Compute a per-split baseline on the same `filtered` set before fitting the
enhanced model, so both are evaluated on identical data:

```python
# inside evaluate_cmf_with_user_attributes, per split
baseline_model = CMF(method="als", k=k, lambda_=lambda_reg, verbose=False)
baseline_model.fit(X=train_df)   # no side information
baseline_result = evaluate_single_split(baseline_model, test_df)
enhanced_result = evaluate_single_split(model, test_df)
improvement = baseline_result["rmse"] - enhanced_result["rmse"]
```

---

## Minor Issues

---

### m-1 — Degree centrality stored as raw count, not normalised

| | |
|---|---|
| **File** | `networks/centrality.py` |
| **Function** | `calculate_degree` |
| **Severity** | MINOR |

**Description**

```python
deg[node.GetId()] = float(node.GetDeg())   # raw degree, not divided by (N-1)
```

All other metrics are naturally in `[0, 1]`; degree values range up to several
hundred.  Although `StandardScaler` downstream partially compensates, the
feature column's pre-scale variance differs by orders of magnitude from the
others.  At minimum the docstring is misleading ("Degree centrality").

**Suggested fix**

```python
n_nodes = G.GetNodes()
denom = max(n_nodes - 1, 1)
deg[node.GetId()] = float(node.GetDeg()) / denom
```

---

### m-2 — Single-user cascades written to the cascade file

| | |
|---|---|
| **File** | `networks/cascades.py` |
| **Function** | `generate_cascades` |
| **Severity** | MINOR |

**Description**

```python
for record in cascades.values():
    if record:   # only skips completely empty records
        fh.write(",".join(map(str, record)) + "\n")
```

Items rated by exactly one user produce a two-element record `[user, ts]` and
are written as a one-node cascade.  These carry no diffusion signal and may
cause edge-case behaviour in NetInf (transmission probability between a single
node and itself is undefined).

**Suggested fix**

```python
if len(record) >= 4:   # at least 2 (user, timestamp) pairs = 4 elements
    fh.write(",".join(map(str, record)) + "\n")
```

---

### m-3 — Dead code: first assignment of `baseline_rmse` is immediately overwritten

| | |
|---|---|
| **File** | `recommender/enhanced.py` |
| **Function** | `run_network_evaluation` |
| **Severity** | MINOR |

**Description**

```python
baseline_rmse = search_result["all_results"][0]["rmse"]  # line A — dead
# use the minimum RMSE across all search iterations as baseline
baseline_rmse = float(min(r["rmse"] for r in search_result["all_results"]))  # line B
```

Line A assigns `all_results[0]["rmse"]` (the first random combination tried,
not the best), then line B immediately overwrites it with the minimum.  Line A
is unreachable dead code and also uses the wrong value if the overwrite were
ever removed.

**Suggested fix**

Delete line A; keep only line B.

---

### m-4 — Unnecessary round-trip through local-timezone `datetime`

| | |
|---|---|
| **File** | `networks/cascades.py` |
| **Function** | `generate_cascades` |
| **Severity** | MINOR |

**Description**

```python
timestamp = datetime.datetime.fromtimestamp(float(row["timestamp"]))  # Unix → local datetime
cascades[item_id].append([user_id, timestamp])
# …
cascades[item_id] = list(itertools.chain(*[[u, dt.timestamp()] for u, dt in records]))
```

`datetime.fromtimestamp` uses the local system timezone.  On hosts where the
timezone is not UTC the intermediate datetime object will be localised, and
`dt.timestamp()` reverses the conversion correctly — but only because the
round-trip cancels out.  The conversion adds fragility and is unnecessary.

**Suggested fix**

Store and write the raw float timestamp directly, avoiding datetime entirely:

```python
cascades[item_id].append([user_id, float(row["timestamp"])])
# sort line:
records.sort(key=lambda x: x[1])  # x[1] is already float
# flatten line:
cascades[item_id] = list(itertools.chain(*records))
```

---

## Summary Table

| ID | File | Function | Severity | Root cause |
|----|------|----------|----------|------------|
| C-1 | `networks/cascades.py` | `generate_cascades` | **CRITICAL** | Cascades sorted descending; NetInf models reversed diffusion direction |
| C-2 | `networks/network_io.py`, `recommender/enhanced.py` | `_build_mapper`, `evaluate_cmf_with_user_attributes` | **CRITICAL** | 1-based compact IDs vs. 0-based LabelEncoder IDs — every user gets wrong side information |
| C-3 | `networks/cascades.py` | `generate_cascades` | **CRITICAL** | Full dataset used for cascades; test interactions contaminate inferred network |
| M-1 | `networks/delta.py` | `compute_median_delta` | MAJOR | All-pairs time differences inflate `Δ` and bias alpha grid centre downward |
| M-2 | `recommender/enhanced.py` | `load_network_features` | MAJOR | Scaler fit on all users before train/test split; test-user statistics leak into training normalisation |
| M-3 | `recommender/enhanced.py` | `run_network_evaluation` | MAJOR | Baseline and enhanced RMSE measured on different user populations; comparison is unfair |
| m-1 | `networks/centrality.py` | `calculate_degree` | MINOR | Raw degree stored instead of normalised degree centrality |
| m-2 | `networks/cascades.py` | `generate_cascades` | MINOR | Single-user cascades written to file; no diffusion signal |
| m-3 | `recommender/enhanced.py` | `run_network_evaluation` | MINOR | Dead code on `baseline_rmse` first assignment |
| m-4 | `networks/cascades.py` | `generate_cascades` | MINOR | Unnecessary UTC-unsafe datetime round-trip |

---

## Recommended Fix Order

1. **C-1** Fix descending sort → re-generate `cascades.txt`.
2. **C-3** Add train/test split before cascade generation → re-generate `cascades.txt`.
3. **C-2** Fix ID offset in `_build_mapper` (0-based) + matching lookup in
   `enhanced.py` → re-run inference, centrality, communities.
4. **M-1** Switch to consecutive delta computation → re-run inference.
5. **M-2** Move scaler fit inside the per-split loop.
6. **M-3** Add a per-split within-subset baseline comparison.

After applying C-1 through C-2 you should observe non-random network edges,
correctly aligned side information, and a meaningful (if modest) RMSE
improvement from the enhanced model.  M-1 through M-3 are required for the
results to be statistically credible.
