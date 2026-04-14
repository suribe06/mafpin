# Pyright Static Analysis Report

**Date:** 2026-04-13  
**Tool:** pyright 1.1.408  
**Environment:** conda env `mafpin`  
**Command:** `python -m pyright .`  
**Result:** 10 errors, 0 warnings, 0 informations

---

## Summary

| File | Errors |
| --- | --- |
| `analysis/shap_analysis.py` | 4 |
| `networks/cascades.py` | 1 |
| `pipeline.py` | 3 |
| `recommender/enhanced.py` | 1 |
| `visualization/community_plots.py` | 1 |

---

## Errors by file

### `analysis/shap_analysis.py` — 4 errors

**Lines 240–241 — `reportAttributeAccessIssue`**

```text
Cannot access attribute "values" for class "ndarray[Any, Unknown]"
Cannot access attribute "values" for class "NDArray[Unknown]"
```

pyright infers the return type of `shap.TreeExplainer.shap_values()` as `ndarray` rather than a `pd.DataFrame`, so it does not recognise the `.values` attribute. The code is correct at runtime; this is a missing/incomplete stub in the `shap` package type annotations.

**Fix:** Cast the result explicitly, or annotate with `# type: ignore[union-attr]`.

---

### `networks/cascades.py` — 1 error

**Line 53 — `reportGeneralTypeIssues`**

```text
Union syntax cannot be used with string operand; use quotes around entire expression
```

A `str | None` union type annotation is written using the Python 3.10+ `X | Y` syntax inside a string annotation context where pyright cannot resolve it.

**Fix:** Use `Optional[str]` from `typing`, or add `from __future__ import annotations` at the top of the file.

---

### `pipeline.py` — 3 errors

**Line 81 — `reportCallIssue` / `reportArgumentType`**

```text
No overloads for "read_csv" match the provided arguments
Argument of type "list[str]" cannot be assigned to parameter "usecols"
```

The pandas stub for `read_csv` has an overly strict `usecols` type (`UsecolsArgType`) that does not accept a plain `list[str]` in older stub versions. The code is correct at runtime.

**Fix:** Cast to `list` or suppress with `# type: ignore[call-overload]`.

**Line 89 — `reportArgumentType`**

```text
Argument of type "Unknown | Any | list[Unknown]" cannot be assigned to parameter
"interactions" of type "DataFrame" in function "generate_cascades_from_df"
```

pyright cannot narrow the type of `train_df` after `train_test_split` because sklearn stubs return a generic tuple. The runtime type is always a `pd.DataFrame`.

**Fix:** Add an explicit `pd.DataFrame` cast after `train_test_split`.

---

### `recommender/enhanced.py` — 1 error

**Line 183 — `reportArgumentType`**

```text
Argument of type "set[Unknown]" cannot be assigned to parameter "values" of type
"Series | DataFrame | Sequence[Unknown] | Mapping[Unknown, Unknown]" in function "isin"
```

`isin()` stubs do not accept `set`; they expect a `Sequence` or `Mapping`. The code is correct at runtime since pandas accepts sets.

**Fix:** Replace `set(user_attributes.index)` with `list(user_attributes.index)`.

---

### `visualization/community_plots.py` — 1 error

**Line 170 — `reportPrivateImportUsage`**

```text
"Axes" is not exported from module "matplotlib.pyplot"
```

`matplotlib.pyplot.Axes` is a re-export that some stub versions mark as private. The correct public import path is `matplotlib.axes.Axes`.

**Fix:** Change `from matplotlib.pyplot import Axes` to `from matplotlib.axes import Axes`.

---

## Classification

| # | Class | Description |
| --- | --- | --- |
| 6 | **Stub gap** | Third-party stubs (shap, pandas, sklearn) do not fully type the used API. Code is correct at runtime. |
| 2 | **Syntax/annotation style** | Python 3.10 union syntax in string annotation context, or missing `from __future__ import annotations`. |
| 2 | **Fixable** | Can be resolved with a minor code change (cast or import path). |

No errors affect the correctness of the pipeline at runtime.
