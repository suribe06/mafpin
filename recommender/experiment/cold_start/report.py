"""Success-criteria report for cold-start hypotheses (§10)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender.experiment.cold_start.paths import ColdStartPaths
from recommender.experiment.cold_start.strata import STRATA_ORDER

# ponytail: below this, stratum-level claims are noise (MovieLens diagnostic 4-10 had N=1)
_MIN_USERS_FOR_CLAIM = 10


def _mean_rmse(results: pd.DataFrame, variant: str, stratum: str) -> float:
    subset = results[
        (results["model_variant"] == variant) & (results["stratum"] == stratum)
    ]
    if subset.empty:
        return float("nan")
    return float(subset["rmse"].iloc[-1])


def _stratum_n_users(results: pd.DataFrame, stratum: str) -> int:
    subset = results[results["stratum"] == stratum]
    if subset.empty or "n_users" not in subset.columns:
        return 0
    return int(subset["n_users"].iloc[-1])


def _stratum_n_ratings(results: pd.DataFrame, stratum: str) -> int:
    subset = results[results["stratum"] == stratum]
    if subset.empty or "n_ratings" not in subset.columns:
        return 0
    return int(subset["n_ratings"].iloc[-1])


def _fmt(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def _bootstrap_section(
    lines: list[str],
    bootstrap: pd.DataFrame,
    *,
    ci_comparisons: tuple[str, ...] = ("M3_vs_M1",),
    ci_strata: tuple[str, ...] = ("1-3", "4-10"),
) -> None:
    lines.extend(
        [
            "",
            "## Bootstrap CIs (per-user mean Δ RMSE)",
            "",
            "Note: table Δ above is **pooled** rating RMSE; "
            "bootstrap is the mean of **per-user** RMSE deltas — signs may differ.",
            "",
            "| Stratum | Comparison | mean | CI low | CI high | N |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.stratum} | {row.comparison} | {_fmt(row.mean_delta)} | "
            f"{_fmt(row.ci_low)} | {_fmt(row.ci_high)} | {int(row.n_users)} |"
        )
    cold_ci = bootstrap[
        (bootstrap["comparison"].isin(ci_comparisons))
        & (bootstrap["stratum"].isin(ci_strata))
        & (bootstrap["n_users"] >= _MIN_USERS_FOR_CLAIM)
    ]
    ci_ok = False
    if not cold_ci.empty:
        ci_ok = all(
            pd.notna(r.ci_low) and float(r.ci_low) > 0
            for r in cold_ci.itertuples(index=False)
        )
    lines.append("")
    lines.append(
        f"**CI check** ({', '.join(ci_comparisons)} on {', '.join(ci_strata)}, "
        f"CI entirely > 0): {'PASS' if ci_ok else 'FAIL/INCONCLUSIVE'}"
    )


def _build_zero_shot_summary(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    *,
    dataset: str,
) -> str:
    """H4 report for trust zero-shot (stratum 0 only)."""
    lines = [
        "# Cold-start success summary",
        "",
        f"- dataset: `{dataset}`",
        "- mode: `zero_shot_trust`",
        "",
        "## RMSE by stratum (trust variants)",
        "",
        "| Estrato | N users | N ratings | M1 | M2_trust | M3_trust | "
        "Δ M2t−M1 | Δ M3t−M1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stratum in STRATA_ORDER:
        m1 = _mean_rmse(results, "M1", stratum)
        m2t = _mean_rmse(results, "M2_trust", stratum)
        m3t = _mean_rmse(results, "M3_trust", stratum)
        n_users = _stratum_n_users(results, stratum)
        n_ratings = _stratum_n_ratings(results, stratum)
        d2 = m1 - m2t if pd.notna(m1) and pd.notna(m2t) else float("nan")
        d3 = m1 - m3t if pd.notna(m1) and pd.notna(m3t) else float("nan")
        lines.append(
            f"| {stratum} | {n_users} | {n_ratings} | {_fmt(m1)} | {_fmt(m2t)} | "
            f"{_fmt(m3t)} | {_fmt(d2)} | {_fmt(d3)} |"
        )

    m1_0 = _mean_rmse(results, "M1", "0")
    m2_0 = _mean_rmse(results, "M2_trust", "0")
    m3_0 = _mean_rmse(results, "M3_trust", "0")
    n0 = _stratum_n_users(results, "0")
    h4_m2 = (
        n0 >= _MIN_USERS_FOR_CLAIM
        and pd.notna(m1_0)
        and pd.notna(m2_0)
        and m2_0 < m1_0
    )
    h4_m3 = (
        n0 >= _MIN_USERS_FOR_CLAIM
        and pd.notna(m1_0)
        and pd.notna(m3_0)
        and m3_0 < m1_0
    )
    lines.extend(
        [
            "",
            "## Criteria (§10 / H4)",
            "",
            f"4a. **H4-M2_trust** (beats M1 on stratum `0`): "
            f"{'PASS' if h4_m2 else 'FAIL/INCONCLUSIVE'} "
            f"(M1={_fmt(m1_0)}, M2_trust={_fmt(m2_0)}, N={n0})",
            f"4b. **H4-M3_trust** (beats M1 on stratum `0`): "
            f"{'PASS' if h4_m3 else 'FAIL/INCONCLUSIVE'} "
            f"(M1={_fmt(m1_0)}, M3_trust={_fmt(m3_0)}, N={n0})",
            "H1/H2 N/A here (all held-out trust users are stratum `0`).",
        ]
    )
    if bootstrap is not None and not bootstrap.empty:
        _bootstrap_section(
            lines,
            bootstrap,
            ci_comparisons=("M2_vs_M1", "M3_vs_M1"),
            ci_strata=("0",),
        )
    else:
        lines.append("")
        lines.append("**CI check**: bootstrap table missing")

    lines.extend(
        [
            "",
            "## Scope note",
            "",
            "Mode `zero_shot_trust` evaluates users with **0** train ratings "
            "using trust-graph attributes (Ciao/Epinions). Separate from NetInf.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_success_summary(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame | None = None,
    *,
    dataset: str,
    mode: str,
) -> str:
    """Render a markdown success summary for H1–H3 style checks."""
    if mode == "zero_shot_trust":
        return _build_zero_shot_summary(results, bootstrap, dataset=dataset)

    lines = [
        "# Cold-start success summary",
        "",
        f"- dataset: `{dataset}`",
        f"- mode: `{mode}`",
        "",
        "## RMSE by stratum",
        "",
        "| Estrato | N users | N ratings | M1 | M2 | M3 | M4c | M4d | Δ M3−M1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    deltas_cold: list[float] = []
    delta_warm = float("nan")
    warnings: list[str] = []

    for stratum in STRATA_ORDER:
        m1 = _mean_rmse(results, "M1", stratum)
        m2 = _mean_rmse(results, "M2", stratum)
        m3 = _mean_rmse(results, "M3", stratum)
        m4c = _mean_rmse(results, "M4c", stratum)
        m4d = _mean_rmse(results, "M4d", stratum)
        n_users = _stratum_n_users(results, stratum)
        n_ratings = _stratum_n_ratings(results, stratum)
        delta = m1 - m3 if pd.notna(m1) and pd.notna(m3) else float("nan")
        if stratum in ("1-3", "4-10"):
            if n_users == 0:
                if mode == "diagnostic":
                    warnings.append(
                        f"Stratum `{stratum}` is empty under global temporal "
                        f"diagnostic; use `--mode controlled` (leave-last or "
                        f"`--split leave_k`)."
                    )
                else:
                    warnings.append(
                        f"Stratum `{stratum}` is empty under `{mode}`. "
                        f"On dense datasets (e.g. MovieLens ≥20 ratings/user) "
                        f"use `--split leave_k`."
                    )
            elif n_users < _MIN_USERS_FOR_CLAIM:
                warnings.append(
                    f"Stratum `{stratum}` has only N={n_users} users "
                    f"(<{_MIN_USERS_FOR_CLAIM}); claims marked inconclusive."
                )
            if pd.notna(delta) and n_users >= _MIN_USERS_FOR_CLAIM:
                deltas_cold.append(delta)
        if stratum == ">10":
            if pd.notna(delta) and n_users >= _MIN_USERS_FOR_CLAIM:
                delta_warm = delta
        lines.append(
            f"| {stratum} | {n_users} | {n_ratings} | {_fmt(m1)} | {_fmt(m2)} | "
            f"{_fmt(m3)} | {_fmt(m4c)} | {_fmt(m4d)} | {_fmt(delta)} |"
        )

    lines.extend(["", "## Criteria (§10)", ""])
    mean_cold = (
        sum(deltas_cold) / len(deltas_cold) if deltas_cold else float("nan")
    )

    # H1a: absolute gain in cold (side info helps cold users).
    h1_gain = pd.notna(mean_cold) and mean_cold > 0
    # H1b: stronger gain in cold than warm (original stricter claim).
    h1_stronger = (
        h1_gain and pd.notna(delta_warm) and mean_cold > delta_warm
    )
    lines.append(
        f"1a. **H1-gain** (M3 beats M1 in cold, Δ>0): "
        f"{'PASS' if h1_gain else 'FAIL/INCONCLUSIVE'} "
        f"(mean Δ cold={_fmt(mean_cold)}; N≥{_MIN_USERS_FOR_CLAIM} only)"
    )
    lines.append(
        f"1b. **H1-stronger** (cold gain > warm gain): "
        f"{'PASS' if h1_stronger else 'FAIL/INCONCLUSIVE'} "
        f"(mean Δ cold={_fmt(mean_cold)}, Δ warm={_fmt(delta_warm)}). "
        f"If 1a PASS and 1b FAIL: side info helps cold users, but not "
        f"*more* than warm — report as general side-info benefit."
    )

    h2_votes: list[bool] = []
    h2_details = []
    for stratum in ("1-3", "4-10"):
        m2 = _mean_rmse(results, "M2", stratum)
        m3 = _mean_rmse(results, "M3", stratum)
        n_users = _stratum_n_users(results, stratum)
        if n_users < _MIN_USERS_FOR_CLAIM or pd.isna(m2) or pd.isna(m3):
            h2_details.append(f"{stratum}=skip(N={n_users})")
            continue
        better = m3 < m2
        h2_votes.append(better)
        h2_details.append(f"{stratum}: M3{'<' if better else '≥'}M2")
    h2_ok = bool(h2_votes) and all(h2_votes)
    lines.append(
        f"2. **H2** (M3 beats M2 in cold): "
        f"{'PASS' if h2_ok else 'FAIL/INCONCLUSIVE'} ({'; '.join(h2_details)})"
    )

    if bootstrap is not None and not bootstrap.empty:
        lines.extend(
            [
                "",
                "## Bootstrap CIs (per-user mean Δ RMSE)",
                "",
                "Note: table Δ above is **pooled** rating RMSE (M1−M3); "
                "bootstrap is the mean of **per-user** RMSE deltas — signs may differ.",
                "",
                "| Stratum | Comparison | mean | CI low | CI high | N |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bootstrap.itertuples(index=False):
            lines.append(
                f"| {row.stratum} | {row.comparison} | {_fmt(row.mean_delta)} | "
                f"{_fmt(row.ci_low)} | {_fmt(row.ci_high)} | {int(row.n_users)} |"
            )
        cold_ci = bootstrap[
            (bootstrap["comparison"] == "M3_vs_M1")
            & (bootstrap["stratum"].isin(["1-3", "4-10"]))
            & (bootstrap["n_users"] >= _MIN_USERS_FOR_CLAIM)
        ]
        ci_ok = False
        if not cold_ci.empty:
            ci_ok = all(
                pd.notna(r.ci_low) and float(r.ci_low) > 0
                for r in cold_ci.itertuples(index=False)
            )
        lines.append("")
        lines.append(
            f"3. **CI check** (cold M3 vs M1 per-user CI entirely > 0): "
            f"{'PASS' if ci_ok else 'FAIL/INCONCLUSIVE'}"
        )
    else:
        lines.append("3. **CI check**: bootstrap table missing")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    if mode == "diagnostic":
        lines.extend(
            [
                "",
                "## Scope note",
                "",
                "Mode `diagnostic` uses the **global temporal** core split. "
                "Stratum sizes can be extreme (many `0`-rating newcomers, few `1-3`/`4-10`). "
                "Authoritative cold-start evidence is `--mode controlled` "
                "(leave-last or `--split leave_k`).",
            ]
        )
    elif mode == "controlled":
        lines.extend(
            [
                "",
                "## Scope note",
                "",
                "Mode `controlled` rebuilds NetInf under `data/<ds>/cold_start/`. "
                "Leave-last keeps natural historial depth; on MovieLens that often "
                "yields only `>10`. Use `--split leave_k` to force cold strata.",
            ]
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def write_success_summary(
    paths: ColdStartPaths,
    *,
    dataset: str,
    mode: str,
    results_path: Path | None = None,
    bootstrap_path: Path | None = None,
) -> Path:
    results_path = results_path or paths.RESULTS
    bootstrap_path = bootstrap_path or paths.BOOTSTRAP_CIS
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results: {results_path}")
    results = pd.read_csv(results_path)
    if "mode" in results.columns:
        results = results[results["mode"] == mode]
    bootstrap = None
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        if "mode" in bootstrap.columns:
            bootstrap = bootstrap[bootstrap["mode"] == mode]
    text = build_success_summary(
        results, bootstrap, dataset=dataset, mode=mode
    )
    out = paths.SUCCESS_SUMMARY
    if mode == "zero_shot_trust":
        out = paths.ZERO_SHOT / "success_summary.md"
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Success summary → {out}")
    return out
