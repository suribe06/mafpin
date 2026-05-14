# Phase 6 Plan: Boundary-Guided Social Regularization

## Current Status

Phases 1 through 5 are already represented in the codebase:

| Phase | Status | Evidence in Code |
| --- | --- | --- |
| 1. Data preparation and cascade construction | Implemented | `pipeline.py` runs `generate_cascades_from_df`; `networks/cascades/generation.py` writes NetInf cascade files from user-item-time data. |
| 2. User-user network inference | Implemented | `networks/inference/core.py` runs NetInf across exponential, power-law, and Rayleigh alpha grids. |
| 3. Overlapping community detection | Implemented | `networks/communities/detection.py` supports Demon and ASLPAw; `networks/communities/batch.py` symmetrizes directed NetInf graphs before community detection. |
| 4. Boundary attributes with h-tilde | Implemented | `networks/communities/lph.py` computes Jaccard LPH, h-tilde, `s_v`, and `delta_v`; `networks/communities/boundary.py` stores `is_boundary`. |
| 5. CMF user attribute integration | Implemented | `recommender/enhanced/features.py` merges centrality and community features; `recommender/enhanced/model.py` passes them as cmfrec `U` side information. |

The saved artifacts also show Phase 5 has been run: the dataset folders contain `cascades.txt`, inferred networks, centrality CSVs, community CSVs, enhanced CMF search results, and SHAP outputs. The SHAP feature list includes centrality and community/boundary variables such as `local_pluralistic_hom`, `lph_score`, `s_v`, `delta_v`, and `is_boundary`.

Phase 6 is not implemented yet. The current enhanced model uses `cmfrec.CMF` with `lambda_`, `w_main`, and `w_user`, which handles side-information reconstruction but does not expose a project-level hook for adding an arbitrary edge-wise graph regularizer of the form needed here. The practical path is to use the local `cmfrec-master/` source tree, patch the C/Cython implementation, and keep the existing cmfrec behavior as the Phase 5 baseline when `lambda_social = 0`.

## Phase 6 Objective

Add a boundary-aware social regularization term to the latent user factors so connected users are not all smoothed equally. The model should still learn from ratings and user-side features, but the inferred network should directly shape the geometry of user embeddings.

Target objective:

$$
\begin{aligned}
\min_{P,\, Q,\, B} \quad
& \sum_{(u,i) \in \Omega} \bigl(r_{ui} - p_u^\top q_i\bigr)^2 \\
& + \alpha_A \|A - PB^\top\|_F^2 \\
& + \lambda \bigl(\|P\|_F^2 + \|Q\|_F^2 + \|B\|_F^2\bigr) \\
& + \lambda_S \sum_{(u,v) \in E} w_{uv}\|p_u - p_v\|^2
\end{aligned}
$$

Where:

- $P$ is the user latent matrix.
- $Q$ is the item latent matrix.
- $A$ is the current user attribute matrix from centrality and community features.
- $B$ maps latent user factors to attributes.
- $E$ comes from the NetInf-inferred user-user network.
- $w_{uv}$ is an adaptive social weight derived from edge structure, overlapping community relation, and boundary scores.
- $\lambda_S$ controls the strength of the social regularizer.

This can also be written as a graph Laplacian penalty:

$$\lambda_S\,\mathrm{tr}(P^\top L_W P)$$

where $L_W = D_W - W$ and $W[u,v] = w_{uv}$.

## Why the C Library Must Be Patched

The current code calls cmfrec like this:

```python
CMF(
    method="als",
    k=k,
    lambda_=lambda_reg,
    w_main=w_main,
    w_user=w_user,
)
```

That is enough for Phase 5 because the boundary metrics enter as columns in the user-side matrix `U`. Phase 6 is different: it needs an additional pairwise penalty across user-user edges that must be evaluated and differentiated jointly over all user latent vectors at every gradient step.

A pure-Python outer loop over edges would reimplement the inner optimization loop in Python and completely undo the C-accelerated performance of cmfrec. The graph Laplacian term $\lambda_S \sum_{(u,v) \in E} w_{uv} \|p_u - p_v\|^2$ must be added directly to the L-BFGS objective and gradient inside the C library.

**Why L-BFGS, not ALS.**
ALS updates each row of `P` independently with a closed-form solve. The graph Laplacian couples all rows of `P` together ($\nabla_{p_u} = \lambda_S L_W p_u$ depends on all neighbours of $u$). Under ALS the per-row system would need to be replaced with a coupled system, which is equivalent to solving a large sparse linear system at each step and is not compatible with the scalar closed-form Cholesky used in cmfrec's ALS path. The L-BFGS path evaluates the objective and gradient globally over all user vectors in one pass, so the Laplacian term integrates naturally.

**Strategy.** Use the local cmfrec source tree in `cmfrec-master/`, build it in place with Cython, and modify three source files:

| File | Change |
| --- | --- |
| `src/cmfrec.h` | New fields in `data_collective_fun_grad`; new parameters in `fit_collective_explicit_lbfgs_internal`. |
| `src/collective.c` | Social regularization block appended in `collective_fun_grad` after the existing L2 block. |
| `cmfrec/wrapper_untyped.pxi` | New parameters in the `cdef extern` declaration and in `call_fit_collective_explicit_lbfgs`. |

The four `.pyx` variant files (`cfuns_double.pyx`, `cfuns_double_plusblas.pyx`, `cfuns_float.pyx`, `cfuns_float_plusblas.pyx`) all `include "wrapper_untyped.pxi"`, so only the `.pxi` file needs to change.

Keep cmfrec for:

- Model 1: baseline CMF.
- Model 2/3: enhanced CMF with side-user attributes.

Use the patched vendor version for:

- Model 4: enhanced CMF plus boundary-guided social regularization.

## Social Weight Design

Start with a simple, testable family of `w_uv` functions. Avoid overloading the first implementation with too many assumptions.

### Variant A: Uniform Social Regularization

Baseline graph regularization:

$$
w_{uv} = 1
$$

Purpose: verify that the custom solver and graph penalty work before adding boundary logic.

### Variant B: Shared-Community Strength

Increase smoothing when users share overlapping communities:

$$
w_{uv} = J(C(u), C(v))
$$

where `J` is Jaccard similarity. Users with no shared community get weak or zero smoothing.

### Variant C: Boundary-Aware Downweighting

Reduce smoothing around boundary users so their latent vectors can preserve cross-community signal:

$$
w_{uv} = J(C(u), C(v)) \cdot g(\tilde{h}_u, \tilde{h}_v)
$$

Suggested first `g`:

$$
g(\tilde{h}_u, \tilde{h}_v) = 1 - \beta \max(b_u, b_v)
$$

where:

- `b_u` is a normalized boundary intensity, high for strongly negative h-tilde.
- `beta` is tuned in `[0, 1]`.

Interpretation: if either endpoint is a boundary user, do not force the pair to be too similar.

### Variant D: Cross-Community Bridge Preservation

Explicitly preserve boundary edges by downweighting edges whose endpoints have low community overlap but high boundary intensity:

$$
w_{uv} = \sigma(\gamma J(C(u), C(v)) - \beta \max(b_u, b_v))
$$

where:

- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid. It keeps $w_{uv}$ between 0 and 1, so the result can be used directly as an edge weight.
- $\gamma$ controls how strongly shared community membership increases the weight. Larger $\gamma$ makes high-overlap users much more likely to be smoothed together.
- $\beta$ controls how strongly boundary intensity decreases the weight. Larger $\beta$ gives boundary users more freedom to keep distinct latent factors.
- $J(C(u), C(v))$ is the Jaccard similarity between the community sets for users $u$ and $v$.

Intuition: the term inside $\sigma$ is a balance between community agreement and boundary tension. If two users share many communities, $\gamma J(C(u), C(v))$ pushes $w_{uv}$ upward. If either user is strongly boundary-like, $\beta \max(b_u, b_v)$ pushes $w_{uv}$ downward. This makes the social regularizer smooth clear within-community relationships while preserving cross-community bridge behavior.

This is more flexible, but should come after Variant C works.

## Proposed Implementation Steps

### Step 1: Add Graph-Regularization Utilities

Create `recommender/enhanced/social_regularization.py` with functions to:

- Load the NetInf network for a `(dataset, model_name, network_index)` pair.
- Symmetrize it with the same strategy used by community detection.
- Load the matching community CSV.
- Build a sparse weighted adjacency matrix `W` aligned to encoded `UserId`.
- Build the sparse Laplacian `L = D - W`.
- Normalize weights so `lambda_S` has comparable meaning across networks.
- Export edges as COO triplets (row, col, val) ready to be passed to C.

Recommended public functions:

```python
def build_social_weights(
    network_file: Path,
    community_features: pd.DataFrame,
    user_index: pd.Index,
    mode: str = "boundary_downweight",
    beta: float = 0.5,
    gamma: float = 1.0,
    symmetrization: str = "union",
) -> scipy.sparse.csr_matrix:
    ...

def graph_laplacian(W: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    ...
```

### Step 2: Build Local cmfrec from Source

```bash
# from the project root
conda run -n mafpin python -m pip uninstall -y cmfrec
cd cmfrec-master && conda run -n mafpin python setup.py build_ext --inplace && cd ..
```

The project keeps `cmfrec-master/` on `sys.path` through the local import bridge, so `from cmfrec import CMF` resolves to `cmfrec-master/cmfrec/` instead of site-packages. The compiled `.so` files are generated directly inside `cmfrec-master/cmfrec/`.

### Step 3: Patch `src/cmfrec.h`

**3a. Add new fields to `data_collective_fun_grad`**

Locate the struct and add before its closing `}`:

```c
/* Social graph Laplacian regularization fields */
int_t   *social_row;     /* COO row indices, upper triangle of W */
int_t   *social_col;     /* COO column indices, upper triangle of W */
real_t  *social_val;     /* COO edge weights */
size_t   social_nnz;     /* number of edges */
real_t   lambda_social;  /* regularization strength */
```

COO format (upper-triangle only) is used here because iterating over edges pair-by-pair is cleaner inside the gradient loop than CSR row iteration.

**3b. Extend `fit_collective_explicit_lbfgs_internal` signature**

Add at the end of the parameter list:

```c
int_t *restrict social_row,
int_t *restrict social_col,
real_t *restrict social_val,
size_t social_nnz,
real_t lambda_social
```

The same parameters must be added to the public `fit_collective_explicit_lbfgs` entry point if it calls `_internal`.

### Step 4: Patch `src/collective.c`

**4a. Populate struct fields in `fit_collective_explicit_lbfgs_internal`**

In the block that fills the `data_collective_fun_grad` struct, after the existing field assignments, add:

```c
data.social_row     = social_row;
data.social_col     = social_col;
data.social_val     = social_val;
data.social_nnz     = social_nnz;
data.lambda_social  = lambda_social;
```

**4b. Add the social gradient block in `collective_fun_grad`**

After the existing L2 regularization block ending with:

```c
taxpy_large(A, lam_unique[2], g_A, (size_t)m_max*(size_t)k_totA, nthreads);
```

Insert:

```c
/* ------------------------------------------------------------------ *
 * Social graph Laplacian regularization                               *
 *   Penalty  = lambda_social * sum_{(u,v) in E} w_uv * ||a_u - a_v||^2
 *   Gradient: grad_A[u] += 2 * lambda_social * w_uv * (a_u - a_v)    *
 *             grad_A[v] += 2 * lambda_social * w_uv * (a_v - a_u)    *
 * Only k+k_main components (rating-relevant dimensions) are          *
 * regularized. k_user components (attribute-only) are left free.     *
 * ------------------------------------------------------------------ */
if (lambda_social != 0. && social_nnz > 0 && social_row != NULL &&
    social_col != NULL && social_val != NULL)
{
    ldouble_safe f_social = 0;
    int_t k_active = k + k_main;
    for (size_t ix = 0; ix < social_nnz; ix++) {
        int_t u = social_row[ix];
        int_t v = social_col[ix];
        real_t wuv = social_val[ix];
        if (u < 0 || v < 0 || u >= m || v >= m || wuv == 0.) continue;

        real_t *restrict Au = A + k_user + (size_t)u * (size_t)k_totA;
        real_t *restrict Av = A + k_user + (size_t)v * (size_t)k_totA;
        real_t *restrict gAu = g_A + k_user + (size_t)u * (size_t)k_totA;
        real_t *restrict gAv = g_A + k_user + (size_t)v * (size_t)k_totA;
        real_t sq_diff = cblas_tdot(k_active, Au, 1, Au, 1)
                         + cblas_tdot(k_active, Av, 1, Av, 1)
                         - 2. * cblas_tdot(k_active, Au, 1, Av, 1);

        f_social += (ldouble_safe)wuv * sq_diff;
        cblas_taxpy(k_active,  2. * lambda_social * wuv, Au, 1, gAu, 1);
        cblas_taxpy(k_active, -2. * lambda_social * wuv, Av, 1, gAu, 1);
        cblas_taxpy(k_active,  2. * lambda_social * wuv, Av, 1, gAv, 1);
        cblas_taxpy(k_active, -2. * lambda_social * wuv, Au, 1, gAv, 1);
    }
    f += (real_t)(lambda_social * f_social);
}
```

**Why `2 * lambda_social` in the gradient?**
The Phase 6 objective stores the social penalty as `lambda_social * sum w_uv ||a_u - a_v||^2`. Differentiating this objective gives `2 * lambda_social * w_uv * (a_u - a_v)` for endpoint `u` and the opposite sign for endpoint `v`, which is the gradient passed to L-BFGS.

**Why `k_active = k + k_main`?**
The latent user vector in cmfrec is partitioned as `[k_user | k | k_main]`. The `k_user` portion is only active in the attribute reconstruction loss `||A - PB||^2`; it does not affect rating predictions. Regularizing it socially would conflate attribute-representation capacity with social smoothing. The rating-relevant components `k + k_main` are what the social penalty should shape.

### Step 5: Patch `cmfrec/wrapper_untyped.pxi`

**5a. Extend the `cdef extern` declaration**

Inside `cdef extern from "cmfrec.h":`, the declaration of `fit_collective_explicit_lbfgs_internal` ends with a list of parameters. Add at the end:

```cython
int_t *social_row,
int_t *social_col,
real_t *social_val,
size_t social_nnz,
real_t lambda_social
```

**5b. Add parameters to `call_fit_collective_explicit_lbfgs`**

In the `def call_fit_collective_explicit_lbfgs(...)` signature, add:

```cython
np.ndarray[int_t, ndim=1] social_row    = np.empty(0, dtype=ctypes.c_int),
np.ndarray[int_t, ndim=1] social_col    = np.empty(0, dtype=ctypes.c_int),
np.ndarray[real_t, ndim=1] social_val   = np.empty(0, dtype=c_real_t),
real_t lambda_social = 0.,
```

#### 5c. Add pointer setup and pass-through

Before the call to `fit_collective_explicit_lbfgs_internal`, add:

```cython
cdef int_t *ptr_social_row = NULL
cdef int_t *ptr_social_col = NULL
cdef real_t *ptr_social_val = NULL
cdef size_t n_social = 0
if social_row.shape[0] > 0 and lambda_social != 0.:
    ptr_social_row = &social_row[0]
    ptr_social_col = &social_col[0]
    ptr_social_val = &social_val[0]
    n_social = <size_t>social_row.shape[0]
```

Add to the argument list passed to `fit_collective_explicit_lbfgs_internal`:

```cython
ptr_social_row, ptr_social_col, ptr_social_val, n_social, lambda_social
```

### Step 6: Recompile

```bash
cd cmfrec-master && conda run -n mafpin python setup.py build_ext --inplace && cd ..
```

Verify the import resolves to the local source copy:

```python
import cmfrec, inspect
print(inspect.getfile(cmfrec))
# should be .../mafpin/cmfrec-master/cmfrec/__init__.py
```

### Step 7: Create `recommender/enhanced/social_model.py`

Wire together the patched library and the social weight builder:

```python
from recommender._cmfrec import CMF  # resolves to cmfrec-master/cmfrec

class SocialCMF:
    """CMF with graph Laplacian social regularization via patched cmfrec."""

    def __init__(self, k=20, lambda_=1.0, w_main=1.0, w_user=0.1,
                 lambda_social=0.01, social_mode="boundary_downweight",
                 beta=0.5, gamma=1.0, random_state=42):
        ...

    def fit(self, ratings_df, user_features_df, social_network_path):
        # 1. Build W from social_network_path and user_features_df
        # 2. Build L_W = D - W as sparse COO (upper triangle)
        # 3. Fit CMF with method="lbfgs", passing social COO arrays
        ...

    def predict(self, user, item):
        ...
```

The `method="lbfgs"` constraint is enforced unconditionally when `lambda_social > 0`. Passing `lambda_social=0` falls back to the unmodified code path and gives an identical result to the unpatched library (since the social block is guarded by `lambda_social != 0.`).

### Step 8: Expose Phase 6 Through the Pipeline

Add a separate pipeline step:

```bash
python pipeline.py --steps social-recommend --include-communities
```

Suggested CLI options:

- `--social-mode`: `uniform`, `community_jaccard`, `boundary_downweight`, `bridge_preserve`.
- `--lambda-social`: graph regularization strength.
- `--boundary-beta`: strength of boundary downweighting.
- `--symmetrization`: `union` or `intersection`.
- `--sample-networks`: reuse existing sampling behavior.

### Step 9: Hyperparameter Search

Extend the enhanced search space only for the social model:

| Parameter | Suggested Range |
| --- | --- |
| `k` | 5 to 50 |
| `lambda_` | log-uniform 0.01 to 10 |
| `w_main` | 0.1 to 1.0 |
| `w_user` | log-uniform 0.01 to 1.0 |
| `lambda_social` | log-uniform 1e-4 to 1.0 |
| `beta` | 0.0 to 1.0 |
| `social_mode` | categorical |

Save results to dataset-scoped files such as:

- `data/<dataset>/social_search_results.json`
- `data/<dataset>/social_regularized_results.json`

### Step 10: Evaluation Ladder

Use the exact comparison ladder from the Next Steps document:

| Model | Implementation |
| --- | --- |
| Model 1 | Existing baseline cmfrec CMF. |
| Model 2 | Existing cmfrec CMF with classical centrality features only. |
| Model 3 | Existing cmfrec enhanced CMF with centrality plus boundary/community features. |
| Model 4a | Patched cmfrec with `lambda_social > 0`, uniform weights. |
| Model 4b | Patched cmfrec with shared-community graph regularization. |
| Model 4c | Patched cmfrec with boundary-aware downweighting. |

Primary metrics:

- RMSE
- MAE
- R2
- NDCG@K
- Precision@K
- Recall@K
- MRR

Structural recommendation metrics to add after the first working version:

- Cross-community exposure.
- Boundary-user recommendation coverage.
- Diversity by community membership.

## Testing Plan

Add focused tests before running full experiments:

1. `build_social_weights` returns a square sparse matrix aligned to `UserId`.
2. Laplacian is symmetric for the undirected graph and has non-negative diagonal.
3. Uniform mode reproduces `w_uv = 1` on graph edges.
4. Boundary mode lowers weights around high-boundary-intensity users.
5. `SocialCMF.predict` returns finite predictions with the same shape as the input arrays.
6. Setting `lambda_social = 0` follows the same objective as the existing enhanced cmfrec model.

## First Milestone

The first implementation milestone should be intentionally narrow:

1. Build the local cmfrec Cython extensions from `cmfrec-master/` and verify imports resolve to the local source tree.
2. Patch the L-BFGS path with the social regularization parameters and guard the block behind `lambda_social != 0`.
3. Run the patched model with `lambda_social = 0` to validate that baseline behavior is preserved.
4. Run it with uniform graph regularization.
5. Add boundary-aware weights only after the uniform regularizer improves or at least behaves stably.

This keeps the research claim clean: first prove that the patched local cmfrec build preserves baseline behavior when the social block is disabled, then test whether h-tilde changes the latent geometry in a useful way.

## Main Risk

The most important risk is not coding complexity; it is experimental attribution. If the patched cmfrec build changes baseline behavior and also adds social regularization at the same time, an improvement could come from either source. That is why `lambda_social = 0` and uniform-social baselines are required before claiming that boundary-guided regularization helped.
