"""Analysis and visualization for Stag Hunt simulation results.

Reads sweep-level CSVs produced by ``sweep_sim.py`` and generates a suite of
publication-ready figures exploring coordination, accuracy, calibration,
influence, and payoffs under adversarial deception.

Usage (CLI)::

    python -m stag_hunt.analysis --logs-dir logs --output-dir output --format png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from stag_hunt.beliefs import compute_belief, compute_q_star, compute_rational_action

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_THEME = "whitegrid"
_MODEL_PALETTE = "Set2"
_ROLE_COLORS = {"Honest": "#4C72B0", "Liar": "#DD8452"}
_LIAR_SHARE_PALETTE = "viridis_r"
_FIG_SINGLE = (8, 5)
_FIG_WIDE = (12, 5)
_FONT_SIZE_BASE = 19
_FONT_SIZE_TICKS = 16
_FONT_SIZE_TITLE = 19
_FONT_SIZE_SMALL = 17
_DENSE_LINE_KWS = {
    "marker": None,
    "linewidth": 1.5,
    "alpha": 0.95,
    "err_kws": {"capsize": 1.2, "elinewidth": 0.8},
}
_DEFAULT_LINE_KWS = {
    "marker": "o",
    "linewidth": 1.8,
    "alpha": 1.0,
    "err_kws": {"capsize": 2, "elinewidth": 1},
}

_LIAR_BIN_ORDER = ["0\u201325%", "25\u201350%", "50\u201375%", "75\u2013100%"]
_BELIEF_MARGIN_BIN_EDGES = np.arange(-0.5, 1.0 + 0.1001, 0.1)
_BENCHMARK_BASELINE_Q = 0.5
_BENCHMARK_MEMORY_LAMBDA = 0.5
_BENCHMARK_PSEUDOCOUNT_TAU = 2.0
_TRUST_INIT_A = 1.0
_TRUST_INIT_B = 0.0
_BENCHMARK_RULE_ORDER = ["Naive aggregate", "Carryover", "Trust-weighted"]
_BENCHMARK_RULE_COLORS = {
    "Naive aggregate": "#1f4e79",
    "Carryover": "#2a9d8f",
    "Trust-weighted": "#7b3f98",
}


def _short_model(name: str) -> str:
    """``'openai/gpt-5-mini'`` -> ``'gpt-5-mini'``."""
    return name.rsplit("/", 1)[-1] if "/" in name else name


def _apply_style() -> None:
    """Apply a consistent visual theme to all figures."""
    sns.set_theme(style=_THEME)
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": _FONT_SIZE_BASE,
            "axes.titlesize": _FONT_SIZE_TITLE,
            "axes.labelsize": _FONT_SIZE_BASE,
            "xtick.labelsize": _FONT_SIZE_TICKS,
            "ytick.labelsize": _FONT_SIZE_TICKS,
            "legend.fontsize": _FONT_SIZE_BASE,
            "legend.title_fontsize": _FONT_SIZE_BASE,
        }
    )


# ---------------------------------------------------------------------------
# Boolean / label helpers
# ---------------------------------------------------------------------------


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a column of ``True``/``False`` (bool *or* str) to boolean."""
    return series.map(
        {True: True, False: False, "True": True, "False": False},
    ).astype(bool)


def _role_label(series: pd.Series) -> pd.Series:
    """Map an ``is_liar`` column to human-readable ``'Liar'``/``'Honest'``."""
    return _as_bool(series).map({True: "Liar", False: "Honest"})


def _fmt_liar_share(share: float) -> str:
    """Format liar share as a percentage string for axis labels."""
    return f"{share:.0%}"


def _bin_liar_share(share: float) -> str:
    """Bin liar share into quartile labels."""
    if share <= 0.25:
        return "0\u201325%"
    elif share <= 0.50:
        return "25\u201350%"
    elif share <= 0.75:
        return "50\u201375%"
    else:
        return "75\u2013100%"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class SweepData:
    """Loaded and enriched DataFrames ready for plotting."""

    runs: pd.DataFrame
    round_metrics: pd.DataFrame
    agent_metrics: pd.DataFrame
    agent_summary: pd.DataFrame
    sweep_points: pd.DataFrame


def _find_csvs(logs_dir: Path, suffix: str) -> list[Path]:
    """Return all CSVs in *logs_dir* whose name ends with *suffix*."""
    return sorted(logs_dir.glob(f"*{suffix}"))


def _read_and_concat(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate multiple CSVs, dropping exact duplicates."""
    if not paths:
        raise FileNotFoundError("No CSV files found for pattern")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def _read_optional_sweep_points(logs: Path) -> pd.DataFrame:
    """Load available sweep-point CSVs (if present) for ablation pairing."""
    paths = sorted(logs.glob("*_sweep_points_all.csv"))
    if not paths:
        paths = sorted(logs.glob("*_sweep_points.csv"))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def load_sweep_data(logs_dir: str | Path) -> SweepData:
    """Load all sweep CSVs from *logs_dir*, enrich columns, and return.

    Enrichments applied:

    * ``model_short`` – last segment of the model identifier.
    * ``liar_share`` – ``num_liars / num_agents`` (ensured on all tables).
    * ``liar_share_label`` – liar share formatted as a percentage string.
    * ``liar_share_bin`` – liar share binned into quartile labels.
    * ``honest_accuracy`` – accuracy of non-liar agents only (round_metrics).
    * ``role`` – ``'Liar'`` / ``'Honest'`` label derived from ``is_liar``.
    * Agent-level tables are joined with run-level parameters
      (``num_agents``, ``num_liars``, ``num_rounds``, ``stag_success_threshold``).
    """
    logs = Path(logs_dir)

    runs = pd.read_csv(logs / "stag_hunt_runs.csv")
    round_metrics = _read_and_concat(_find_csvs(logs, "_round_metrics.csv"))
    agent_metrics = _read_and_concat(_find_csvs(logs, "_agent_metrics.csv"))
    agent_summary = _read_and_concat(_find_csvs(logs, "_agent_summary.csv"))
    sweep_points = _read_optional_sweep_points(logs)

    # Short model names for plot labels
    for df in (runs, round_metrics, agent_metrics, agent_summary):
        if "model" in df.columns:
            df["model_short"] = df["model"].apply(_short_model)

    # Ensure liar_share on runs and round_metrics
    for df in (runs, round_metrics):
        if "liar_share" not in df.columns:
            df["liar_share"] = df["num_liars"] / df["num_agents"]

    # Compute honest_accuracy: accuracy of non-liar agents only.
    # honest_correct = total_correct - liar_correct
    #                = round_accuracy * num_agents - liar_accuracy * num_liars
    num_honest = round_metrics["num_agents"] - round_metrics["num_liars"]
    honest_correct = (
        round_metrics["round_accuracy"] * round_metrics["num_agents"]
        - round_metrics["liar_accuracy"] * round_metrics["num_liars"]
    )
    round_metrics["honest_accuracy"] = honest_correct / num_honest.replace(
        0, float("nan")
    )

    # Enrich agent and round tables with run-level parameters
    run_params = runs[
        [
            "run_id",
            "num_agents",
            "num_liars",
            "num_rounds",
            "stag_success_threshold",
        ]
    ].drop_duplicates()
    agent_summary = agent_summary.merge(run_params, on="run_id", how="left")
    agent_metrics = agent_metrics.merge(run_params, on="run_id", how="left")

    for df in (agent_metrics, agent_summary):
        if "liar_share" not in df.columns:
            df["liar_share"] = df["num_liars"] / df["num_agents"]

    # Coerce boolean-ish columns
    _bool_cols = ("stag_success", "is_liar", "is_correct", "was_flipped")
    for col in _bool_cols:
        for df in (round_metrics, agent_metrics):
            if col in df.columns:
                df[col] = _as_bool(df[col])

    # Role labels on agent tables
    for df in (agent_metrics, agent_summary):
        if "is_liar" in df.columns:
            df["role"] = _role_label(df["is_liar"])

    # Liar share labels (formatted %) and quartile bins on all tables
    for df in (runs, round_metrics, agent_metrics, agent_summary):
        if "liar_share" in df.columns:
            df["liar_share_label"] = df["liar_share"].apply(_fmt_liar_share)
            df["liar_share_bin"] = df["liar_share"].apply(_bin_liar_share)

    return SweepData(
        runs=runs,
        round_metrics=round_metrics,
        agent_metrics=agent_metrics,
        agent_summary=agent_summary,
        sweep_points=sweep_points,
    )


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------


def _multi_round_filter(rm: pd.DataFrame) -> pd.DataFrame:
    """Return only rows belonging to runs that played more than one round."""
    max_round = rm.groupby("run_id")["round"].max()
    multi_run_ids = max_round[max_round > 1].index
    return rm[rm["run_id"].isin(multi_run_ids)].copy()


def _sorted_models(df: pd.DataFrame) -> list[str]:
    """Sorted unique ``model_short`` values."""
    return sorted(df["model_short"].unique())


def _model_color_map(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Stable color mapping for all models present in *df*."""
    models = _sorted_models(df)
    palette = sns.color_palette(_MODEL_PALETTE, n_colors=len(models))
    return dict(zip(models, palette))


def _sorted_liar_labels(df: pd.DataFrame) -> list[str]:
    """Return unique liar_share_label values sorted by numeric value."""
    return sorted(
        df["liar_share_label"].unique(),
        key=lambda x: float(x.strip("%")) / 100,
    )


def _empty_fig(message: str) -> plt.Figure:
    """Return a single-axis figure with a centred text message."""
    fig, ax = plt.subplots(figsize=_FIG_SINGLE)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _add_figure_legend(fig: plt.Figure, ax: plt.Axes, **kwargs) -> None:
    """Extract handles/labels from *ax* and attach a single figure legend."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, **kwargs)


def _lineplot_with_errorbars(
    *,
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    hue_order: list[str],
    palette: str | dict[str, str],
    errorbar: tuple[str, int],
    sort: bool = True,
    dense: bool = False,
) -> None:
    """Consistent lineplot styling with error bars instead of filled bands."""
    style_kws = _DENSE_LINE_KWS if dense else _DEFAULT_LINE_KWS
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        marker=style_kws["marker"],
        linewidth=style_kws["linewidth"],
        alpha=style_kws["alpha"],
        palette=palette,
        errorbar=errorbar,
        err_style="bars",
        err_kws=style_kws["err_kws"],
        sort=sort,
        ax=ax,
    )


def _base_only_data(data: SweepData) -> SweepData:
    """Return a copy of *data* restricted to default (non-ablation) runs."""
    runs = data.runs.copy()

    default_checks = []
    if "order_ablation" in runs.columns:
        default_checks.append(runs["order_ablation"] == "a1")
    if "adversary_ablation" in runs.columns:
        default_checks.append(runs["adversary_ablation"] == "base")
    if "heterogeneity_ablation" in runs.columns:
        default_checks.append(runs["heterogeneity_ablation"] == "h1")

    if default_checks:
        base_mask = default_checks[0]
        for cond in default_checks[1:]:
            base_mask = base_mask & cond
        base_runs = runs[base_mask].copy()
    else:
        base_runs = runs.copy()

    base_run_ids = set(base_runs["run_id"])
    round_metrics = data.round_metrics[
        data.round_metrics["run_id"].isin(base_run_ids)
    ].copy()
    agent_metrics = data.agent_metrics[
        data.agent_metrics["run_id"].isin(base_run_ids)
    ].copy()
    agent_summary = data.agent_summary[
        data.agent_summary["run_id"].isin(base_run_ids)
    ].copy()

    sweep_points = data.sweep_points.copy()
    if not sweep_points.empty and "ablation_code" in sweep_points.columns:
        sweep_points = sweep_points[sweep_points["ablation_code"] == "base"].copy()

    return SweepData(
        runs=base_runs,
        round_metrics=round_metrics,
        agent_metrics=agent_metrics,
        agent_summary=agent_summary,
        sweep_points=sweep_points,
    )


# ---------------------------------------------------------------------------
# Public-belief benchmark reconstruction
# ---------------------------------------------------------------------------


def build_belief_benchmark(data: SweepData) -> pd.DataFrame:
    """Reconstruct per-turn public-belief benchmark quantities from sweep CSVs."""
    am = data.agent_metrics.copy()
    if am.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "round",
                "turn_index",
                "agent",
                "model_short",
                "liar_share_bin",
                "role",
                "alpha",
                "k_stag_seen",
                "n_observed",
                "q_hat",
                "q_star",
                "q_margin",
                "rational_action",
                "benchmark_defined",
                "matches_benchmark",
                "false_cooperate",
                "false_defect",
            ]
        )

    candidate_run_cols = [
        "payoff_stag_success",
        "payoff_stag_fail",
        "payoff_hare_fail",
        "stag_success_threshold",
    ]
    missing_run_cols = [col for col in candidate_run_cols if col not in am.columns]
    bench = am.copy()
    if missing_run_cols:
        run_params = data.runs[["run_id", *missing_run_cols]].drop_duplicates()
        bench = bench.merge(run_params, on="run_id", how="left", validate="many_to_one")

    for col in (
        "reported_is_stag",
        "original_is_stag",
        "is_correct",
        "is_liar",
        "stag_success",
    ):
        if col in bench.columns:
            bench[col] = _as_bool(bench[col])

    if "role" not in bench.columns and "is_liar" in bench.columns:
        bench["role"] = _role_label(bench["is_liar"])
    if "model_short" not in bench.columns and "model" in bench.columns:
        bench["model_short"] = bench["model"].apply(_short_model)
    if "liar_share" not in bench.columns:
        bench["liar_share"] = bench["num_liars"] / bench["num_agents"]
    if "liar_share_bin" not in bench.columns:
        bench["liar_share_bin"] = bench["liar_share"].apply(_bin_liar_share)

    bench = bench.sort_values(["run_id", "round", "turn_index", "agent"]).copy()
    bench["k_stag_seen"] = (
        bench.groupby(["run_id", "round"], sort=False)["reported_is_stag"]
        .transform(lambda s: s.astype(int).cumsum().shift(fill_value=0))
        .astype(int)
    )
    bench["n_observed"] = bench["turn_index"].astype(int)

    q_star_cache: dict[tuple[int, int, float, float, float], float] = {}
    alpha_vals: list[float] = []
    q_hat_vals: list[float | None] = []
    q_star_vals: list[float] = []
    rational_vals: list[str | None] = []

    for row in bench.itertuples(index=False):
        alpha, q_hat = compute_belief(
            k_stag=int(row.k_stag_seen),
            n_observed=int(row.n_observed),
            num_agents=int(row.num_agents),
            num_liars=int(row.num_liars),
        )
        q_key = (
            int(row.num_agents),
            int(row.stag_success_threshold),
            float(row.payoff_stag_success),
            float(row.payoff_stag_fail),
            float(row.payoff_hare_fail),
        )
        q_star = q_star_cache.get(q_key)
        if q_star is None:
            q_star = compute_q_star(
                num_agents=q_key[0],
                threshold_m=q_key[1],
                payoff_stag_success=q_key[2],
                payoff_stag_fail=q_key[3],
                payoff_hare_safe=q_key[4],
            )
            q_star_cache[q_key] = q_star

        rational_action = None
        if q_hat is not None:
            rational_action = compute_rational_action(
                q=q_hat,
                num_agents=int(row.num_agents),
                threshold_m=int(row.stag_success_threshold),
                payoff_stag_success=float(row.payoff_stag_success),
                payoff_stag_fail=float(row.payoff_stag_fail),
                payoff_hare_safe=float(row.payoff_hare_fail),
            )

        alpha_vals.append(alpha)
        q_hat_vals.append(q_hat)
        q_star_vals.append(q_star)
        rational_vals.append(rational_action)

    bench["alpha"] = alpha_vals
    bench["q_hat"] = q_hat_vals
    bench["q_star"] = q_star_vals
    bench["q_margin"] = bench["q_hat"] - bench["q_star"]
    bench["rational_action"] = rational_vals
    bench["benchmark_defined"] = bench["q_hat"].notna()
    bench["matches_benchmark"] = bench["benchmark_defined"] & (
        bench["original_action"] == bench["rational_action"]
    )
    bench["false_cooperate"] = (
        bench["benchmark_defined"]
        & (bench["q_margin"] < 0)
        & (bench["original_action"] == "STAG")
    )
    bench["false_defect"] = (
        bench["benchmark_defined"]
        & (bench["q_margin"] > 0)
        & (bench["original_action"] == "HARE")
    )
    bench["q_margin_clipped"] = bench["q_margin"].clip(
        lower=_BELIEF_MARGIN_BIN_EDGES[0],
        upper=_BELIEF_MARGIN_BIN_EDGES[-1],
    )
    bench["q_margin_bin"] = pd.cut(
        bench["q_margin_clipped"],
        bins=_BELIEF_MARGIN_BIN_EDGES,
        include_lowest=True,
        duplicates="drop",
    )
    bench["q_margin_mid"] = bench["q_margin_bin"].map(
        lambda interval: interval.mid if pd.notna(interval) else np.nan
    )
    bench["benchmark_rule"] = "Naive aggregate"
    bench["benchmark_q"] = bench["q_hat"]
    return bench


def _finalize_benchmark_rule(
    bench: pd.DataFrame,
    *,
    q_values: pd.Series,
    rule_name: str,
) -> pd.DataFrame:
    """Attach rule-specific benchmark quantities to a prepared benchmark frame."""
    out = bench.copy()
    out["benchmark_rule"] = rule_name
    out["benchmark_q"] = q_values.reindex(out.index)
    out["q_margin"] = out["benchmark_q"] - out["q_star"]

    rational_vals: list[str | None] = []
    for row in out.itertuples(index=False):
        rational_action = None
        if pd.notna(row.benchmark_q):
            rational_action = compute_rational_action(
                q=float(row.benchmark_q),
                num_agents=int(row.num_agents),
                threshold_m=int(row.stag_success_threshold),
                payoff_stag_success=float(row.payoff_stag_success),
                payoff_stag_fail=float(row.payoff_stag_fail),
                payoff_hare_safe=float(row.payoff_hare_fail),
            )
        rational_vals.append(rational_action)

    out["rational_action"] = rational_vals
    out["benchmark_defined"] = out["benchmark_q"].notna()
    out["matches_benchmark"] = out["benchmark_defined"] & (
        out["original_action"] == out["rational_action"]
    )
    out["false_cooperate"] = (
        out["benchmark_defined"]
        & (out["q_margin"] < 0)
        & (out["original_action"] == "STAG")
    )
    out["false_defect"] = (
        out["benchmark_defined"]
        & (out["q_margin"] > 0)
        & (out["original_action"] == "HARE")
    )
    out["q_margin_clipped"] = out["q_margin"].clip(
        lower=_BELIEF_MARGIN_BIN_EDGES[0],
        upper=_BELIEF_MARGIN_BIN_EDGES[-1],
    )
    out["q_margin_bin"] = pd.cut(
        out["q_margin_clipped"],
        bins=_BELIEF_MARGIN_BIN_EDGES,
        include_lowest=True,
        duplicates="drop",
    )
    out["q_margin_mid"] = out["q_margin_bin"].map(
        lambda interval: interval.mid if pd.notna(interval) else np.nan
    )
    return out


def build_carryover_benchmark(data: SweepData) -> pd.DataFrame:
    """Benchmark using a carryover prior from the previous round."""
    bench = build_belief_benchmark(data)
    if bench.empty:
        return bench

    q_values = pd.Series(index=bench.index, dtype=float)
    for _, run_df in bench.groupby("run_id", sort=False):
        run_df = run_df.sort_values(["round", "turn_index", "agent"])
        prev_post = _BENCHMARK_BASELINE_Q
        first_round = True

        for _, round_df in run_df.groupby("round", sort=True):
            prior = (
                _BENCHMARK_BASELINE_Q
                if first_round
                else _BENCHMARK_MEMORY_LAMBDA * prev_post
                + (1 - _BENCHMARK_MEMORY_LAMBDA) * _BENCHMARK_BASELINE_Q
            )
            for row in round_df.itertuples():
                denom = _BENCHMARK_PSEUDOCOUNT_TAU + float(row.n_observed)
                q_values.at[row.Index] = (
                    _BENCHMARK_PSEUDOCOUNT_TAU * prior + float(row.k_stag_seen)
                ) / denom

            total_stag = float(round_df["reported_is_stag"].astype(int).sum())
            prev_post = (_BENCHMARK_PSEUDOCOUNT_TAU * prior + total_stag) / (
                _BENCHMARK_PSEUDOCOUNT_TAU + len(round_df)
            )
            first_round = False

    return _finalize_benchmark_rule(bench, q_values=q_values, rule_name="Carryover")


def build_trust_weighted_benchmark(data: SweepData) -> pd.DataFrame:
    """Benchmark using agent-specific trust weights updated across rounds."""
    bench = build_belief_benchmark(data)
    if bench.empty:
        return bench

    q_values = pd.Series(index=bench.index, dtype=float)
    for _, run_df in bench.groupby("run_id", sort=False):
        run_df = run_df.sort_values(["round", "turn_index", "agent"])
        agents = list(dict.fromkeys(run_df["agent"].tolist()))
        trust_a = {agent: _TRUST_INIT_A for agent in agents}
        trust_b = {agent: _TRUST_INIT_B for agent in agents}
        prev_post = _BENCHMARK_BASELINE_Q
        first_round = True

        for _, round_df in run_df.groupby("round", sort=True):
            prior = (
                _BENCHMARK_BASELINE_Q
                if first_round
                else _BENCHMARK_MEMORY_LAMBDA * prev_post
                + (1 - _BENCHMARK_MEMORY_LAMBDA) * _BENCHMARK_BASELINE_Q
            )
            seen_weight = 0.0
            seen_weighted_stag = 0.0

            for row in round_df.itertuples():
                denom = _BENCHMARK_PSEUDOCOUNT_TAU + seen_weight
                q_values.at[row.Index] = (
                    _BENCHMARK_PSEUDOCOUNT_TAU * prior + seen_weighted_stag
                ) / denom

                rho = trust_a[row.agent] / (trust_a[row.agent] + trust_b[row.agent])
                seen_weight += rho
                seen_weighted_stag += rho * float(row.reported_is_stag)

            prev_post = (_BENCHMARK_PSEUDOCOUNT_TAU * prior + seen_weighted_stag) / (
                _BENCHMARK_PSEUDOCOUNT_TAU + seen_weight
            )

            for row in round_df.itertuples():
                z = 1.0 if bool(row.reported_is_stag) == bool(row.stag_success) else 0.0
                trust_a[row.agent] += z
                trust_b[row.agent] += 1.0 - z

            first_round = False

    return _finalize_benchmark_rule(
        bench, q_values=q_values, rule_name="Trust-weighted"
    )


def _fig_honest_benchmark_response(
    bench: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    x_label: str,
    color: str,
    empty_message: str,
) -> plt.Figure:
    """Shared response-curve figure for honest-agent benchmark comparisons."""
    if bench.empty:
        return _empty_fig(empty_message)

    honest = bench[(bench["role"] == "Honest") & bench["benchmark_defined"]].copy()
    honest = honest.dropna(subset=["q_margin_mid"]).copy()
    if honest.empty:
        return _empty_fig("No eligible honest-agent benchmark rows found")

    honest[y_col] = _as_bool(honest[y_col])
    models = _sorted_models(honest)
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    counts = (
        honest.groupby(["model_short", "q_margin_mid"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    honest = honest.merge(counts, on=["model_short", "q_margin_mid"], how="left")
    honest = honest[honest["n"] >= 8].copy()
    if honest.empty:
        return _empty_fig("No sufficiently populated honest-agent benchmark bins found")

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = honest[honest["model_short"] == model].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="q_margin_mid",
            y=y_col,
            hue="model_short",
            hue_order=[model],
            palette={model: color},
            errorbar=("ci", 95),
            dense=True,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(model)
        ax.set_xlabel(x_label if row == n_rows - 1 else "")
        ax.set_ylabel(ylabel if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(
            _BELIEF_MARGIN_BIN_EDGES[0] - 0.05, _BELIEF_MARGIN_BIN_EDGES[-1] + 0.05
        )
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    if not visible_axes:
        return _empty_fig("No eligible honest-agent benchmark rows found")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1 – Coordination success rate vs. liar fraction
# ---------------------------------------------------------------------------


def fig_coordination_vs_liar_share(data: SweepData) -> plt.Figure:
    """Line plot of stag-success rate vs. liar fraction, by model.

    Rows = stag_success_threshold, columns = num_agents.
    """
    rm = data.round_metrics

    # One observation per run (aggregate across rounds)
    run_agg = (
        rm.groupby(
            [
                "run_id",
                "model_short",
                "num_agents",
                "liar_share",
                "stag_success_threshold",
            ]
        )
        .agg(stag_success_rate=("stag_success", "mean"))
        .reset_index()
    )

    model_colors = _model_color_map(data.runs)
    models = [m for m in _sorted_models(run_agg) if m in model_colors]
    all_thresholds = sorted(run_agg["stag_success_threshold"].unique())
    # Drop M=1 — success is trivially near-1.0 when only one agent must choose
    # stag, making those panels uninformative. Fall back to all if none remain.
    thresholds = [t for t in all_thresholds if t > 1] or all_thresholds

    pairs = sorted(
        {
            (int(n_agents), int(threshold))
            for n_agents, threshold in run_agg[
                ["num_agents", "stag_success_threshold"]
            ].itertuples(index=False, name=None)
            if threshold in thresholds
        }
    )
    if not pairs:
        return _empty_fig("No non-trivial coordination settings found")

    n_cols = min(3, len(pairs))
    n_rows = math.ceil(len(pairs) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for ax in flat_axes[len(pairs) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, (n_agents, threshold) in enumerate(pairs):
        ax = flat_axes[idx]
        subset = run_agg[
            (run_agg["num_agents"] == n_agents)
            & (run_agg["stag_success_threshold"] == threshold)
        ].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        subset = subset.sort_values("liar_share")

        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="liar_share",
            y="stag_success_rate",
            hue="model_short",
            hue_order=models,
            palette=model_colors,
            errorbar=("ci", 95),
            sort=False,
        )

        # Add vertical cutoff line at 1 - m/n
        cutoff = 1 - threshold / n_agents
        ax.axvline(
            x=cutoff,
            linestyle="--",
            color="black",
            alpha=0.6,
            label="(1 - M/N) threshold",
        )
        row, col = divmod(idx, n_cols)
        ax.set_title(f"N={n_agents}, M={threshold}")
        ax.set_xlabel("Liar fraction" if row == n_rows - 1 else "")
        ax.set_ylabel("Stag success rate" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=0)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    first_visible = visible_axes[0]
    _add_figure_legend(
        fig,
        first_visible,
        title="Model",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def fig1_highlight(data: SweepData) -> plt.Figure:
    """Figure 1 highlight: coordination vs liar fraction for N=5, M=3 only."""
    rm = data.round_metrics

    run_agg = (
        rm.groupby(
            [
                "run_id",
                "model_short",
                "num_agents",
                "liar_share",
                "stag_success_threshold",
            ]
        )
        .agg(stag_success_rate=("stag_success", "mean"))
        .reset_index()
    )

    subset = run_agg[
        (run_agg["num_agents"] == 5) & (run_agg["stag_success_threshold"] == 3)
    ].copy()
    if subset.empty:
        return _empty_fig("No data found for highlight setting N=5, M=3")

    subset = subset.sort_values("liar_share")
    model_colors = _model_color_map(data.runs)
    models = [m for m in _sorted_models(subset) if m in model_colors]

    fig, ax = plt.subplots(1, 1, figsize=_FIG_SINGLE)
    _lineplot_with_errorbars(
        ax=ax,
        data=subset,
        x="liar_share",
        y="stag_success_rate",
        hue="model_short",
        hue_order=models,
        palette=model_colors,
        errorbar=("ci", 95),
        sort=False,
    )

    ax.axvline(
        x=(1 - 3 / 5),
        linestyle="--",
        color="black",
        alpha=0.6,
        label="(1 - M/N) threshold",
    )
    ax.set_title("N=5, M=3")
    ax.set_xlabel("Liar fraction")
    ax.set_ylabel("Stag success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if ax.get_legend():
        ax.get_legend().remove()
    _add_figure_legend(
        fig,
        ax,
        title="Model",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – Accuracy trajectory over rounds
# ---------------------------------------------------------------------------


def fig_accuracy_over_rounds(data: SweepData) -> plt.Figure:
    """Honest-agent accuracy and confidence over rounds, faceted by model.

    Top row = accuracy, bottom row = confidence.  Hue = liar-fraction bin
    (quartile) so the CI bands stay readable.
    """
    rm = _multi_round_filter(data.round_metrics)
    if rm.empty:
        return _empty_fig("No multi-round runs found")

    models = _sorted_models(rm)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in rm["liar_share_bin"].values]

    metrics = [
        ("honest_accuracy", "Honest-agent accuracy"),
        ("confidence_mean", "Mean confidence"),
    ]
    n_rows = len(metrics)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, model in enumerate(models):
            ax = axes[row, col]
            subset = rm[rm["model_short"] == model]
            _lineplot_with_errorbars(
                ax=ax,
                data=subset,
                x="round",
                y=metric_col,
                hue="liar_share_bin",
                hue_order=bin_order,
                palette=_LIAR_SHARE_PALETTE,
                errorbar=("ci", 95),
                dense=True,
            )
            if row == 0:
                ax.set_title(model)
            ax.set_xlabel("Round" if row == n_rows - 1 else "")
            ax.set_ylabel(metric_label if col == 0 else "")
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if ax.get_legend():
                ax.get_legend().remove()

    # Zoom confidence row: values cluster in upper range so the full [0,1]
    # axis wastes most of the space.
    conf_col = rm["confidence_mean"].dropna()
    if not conf_col.empty and len(metrics) > 1:
        conf_min = max(0.0, float(conf_col.quantile(0.01)) - 0.05)
        axes[1, 0].set_ylim(conf_min, 1.02)

    _add_figure_legend(
        fig,
        axes[0, 0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 (alternate) – Combined models on one figure
# ---------------------------------------------------------------------------


def fig2_alternate(data: SweepData) -> plt.Figure:
    """Combined-model view of Figure 2.

    Rows = metrics (accuracy, confidence), columns = liar-fraction bins.
    Each panel overlays all models so cross-model comparisons are direct.
    """
    rm = _multi_round_filter(data.round_metrics)
    if rm.empty:
        return _empty_fig("No multi-round runs found")

    model_colors = _model_color_map(data.runs)
    models = [m for m in _sorted_models(rm) if m in model_colors]
    bin_order = [b for b in _LIAR_BIN_ORDER if b in rm["liar_share_bin"].values]
    if not bin_order:
        return _empty_fig("No liar-fraction bins available")

    metrics = [
        ("honest_accuracy", "Honest-agent accuracy"),
        ("confidence_mean", "Mean confidence"),
    ]
    n_rows = len(metrics)
    n_cols = len(bin_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, bin_label in enumerate(bin_order):
            ax = axes[row, col]
            subset = rm[rm["liar_share_bin"] == bin_label]
            _lineplot_with_errorbars(
                ax=ax,
                data=subset,
                x="round",
                y=metric_col,
                hue="model_short",
                hue_order=models,
                palette=model_colors,
                errorbar=("ci", 95),
                dense=True,
            )
            if row == 0:
                ax.set_title(f"Liar fraction {bin_label}")
            ax.set_xlabel("Round" if row == n_rows - 1 else "")
            ax.set_ylabel(metric_label if col == 0 else "")
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if ax.get_legend():
                ax.get_legend().remove()

    # Zoom confidence row: values cluster in upper range, so a full [0,1]
    # axis wastes space and masks model separation.
    conf_col = rm["confidence_mean"].dropna()
    if not conf_col.empty and len(metrics) > 1:
        conf_min = max(0.0, float(conf_col.quantile(0.01)) - 0.05)
        axes[1, 0].set_ylim(conf_min, 1.02)

    _add_figure_legend(
        fig,
        axes[0, 0],
        title="Model",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def fig2_alternate_b(data: SweepData) -> plt.Figure:
    """Focused combined-model view emphasising confident error.

    This variant keeps all liar-fraction bins for consistency with the rest of
    the paper, but visually de-emphasises per-model traces by pushing them into
    the background and highlighting the across-model mean.
    """
    rm = _multi_round_filter(data.round_metrics)
    if rm.empty:
        return _empty_fig("No multi-round runs found")

    am = data.agent_metrics.copy()
    if am.empty or "confidence" not in am.columns or "is_liar" not in am.columns:
        return _empty_fig("No agent-level confidence data found")
    am = am[~_as_bool(am["is_liar"])].copy()
    if am.empty:
        return _empty_fig("No honest-agent confidence data found")

    honest_conf = (
        am.groupby(["run_id", "round"], observed=False)
        .agg(honest_confidence_mean=("confidence", "mean"))
        .reset_index()
    )
    rm = rm.copy().merge(honest_conf, on=["run_id", "round"], how="left")
    rm = rm.dropna(subset=["honest_confidence_mean"]).copy()
    if rm.empty:
        return _empty_fig("No matched honest-confidence data found")
    rm["confidence_gap"] = rm["honest_confidence_mean"] - rm["honest_accuracy"]

    models = _sorted_models(rm)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in rm["liar_share_bin"].values]
    if not bin_order:
        return _empty_fig("No liar-fraction bins available")

    metrics = [
        ("honest_accuracy", "Honest-agent accuracy"),
        ("confidence_gap", "Honest confidence - accuracy"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(bin_order),
        figsize=(4.6 * len(bin_order), 4 * len(metrics)),
        sharey="row",
        squeeze=False,
    )
    row_mins = [float("inf")] * len(metrics)
    row_maxs = [float("-inf")] * len(metrics)

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, bin_label in enumerate(bin_order):
            ax = axes[row, col]
            subset = rm[
                (rm["liar_share_bin"] == bin_label) & (rm["model_short"].isin(models))
            ]
            if subset.empty:
                ax.set_visible(False)
                continue

            per_model = (
                subset.groupby(["model_short", "round"], observed=False)[metric_col]
                .mean()
                .reset_index()
            )
            mean_trend = (
                per_model.groupby("round", observed=False)[metric_col]
                .mean()
                .reset_index()
            )

            # Show individual model trajectories as faint background context.
            for model in models:
                model_df = per_model[per_model["model_short"] == model].sort_values(
                    "round"
                )
                if model_df.empty:
                    continue
                ax.plot(
                    model_df["round"],
                    model_df[metric_col],
                    color="0.70",
                    linewidth=1.2,
                    alpha=0.55,
                    zorder=1,
                )

            # Foreground the cross-model mean using the same CI machinery as the
            # main line figures: seaborn mean estimate with 95% CI bars.
            sns.lineplot(
                data=per_model,
                x="round",
                y=metric_col,
                errorbar=("ci", 95),
                err_style="bars",
                color="#1f4e79",
                err_kws={"capsize": 2.5, "elinewidth": 1.2},
                linewidth=2.6,
                marker="o",
                markersize=4.5,
                zorder=3,
                ax=ax,
            )

            local_min = min(
                float(per_model[metric_col].min()),
                float(mean_trend[metric_col].min()),
            )
            local_max = max(
                float(per_model[metric_col].max()),
                float(mean_trend[metric_col].max()),
            )
            row_mins[row] = min(row_mins[row], local_min)
            row_maxs[row] = max(row_maxs[row], local_max)

            if row == 0:
                ax.set_title(f"{bin_label} liars")
            ax.set_xlabel("Round" if row == len(metrics) - 1 else "")
            ax.set_ylabel(metric_label if col == 0 else "")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if row == 1:
                ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1, alpha=0.8)
            if ax.get_legend():
                ax.get_legend().remove()

    for row, (metric_col, _) in enumerate(metrics):
        if not np.isfinite(row_mins[row]) or not np.isfinite(row_maxs[row]):
            continue
        padding = max(0.03, 0.08 * (row_maxs[row] - row_mins[row]))
        y_min = row_mins[row] - padding
        y_max = row_maxs[row] + padding
        if metric_col == "honest_accuracy":
            y_min = max(0.0, y_min)
            y_max = min(1.02, y_max)
        for col in range(len(bin_order)):
            axes[row, col].set_ylim(y_min, y_max)

    fig.legend(
        handles=[
            Line2D([0], [0], color="#1f4e79", marker="o", linewidth=2.6),
            Line2D([0], [0], color="0.70", linewidth=1.2, alpha=0.55),
        ],
        labels=["Across-model mean", "Individual model"],
        title="Series",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Confidence calibration
# ---------------------------------------------------------------------------


def fig_confidence_calibration(data: SweepData) -> plt.Figure:
    """Reliability-style calibration plot (binned confidence vs. accuracy)."""
    ags = data.agent_summary.dropna(subset=["confidence_mean", "accuracy"])
    if ags.empty:
        return _empty_fig("No calibration data found")

    ags = ags.copy()
    ags["confidence_mean"] = ags["confidence_mean"].clip(0, 1)
    ags["accuracy"] = ags["accuracy"].clip(0, 1)
    # Equal-width bins tame overplotting in the raw discrete scatter.
    bin_edges = np.linspace(0, 1, 9)
    ags["confidence_bin"] = pd.cut(
        ags["confidence_mean"],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    cal = (
        ags.groupby(["role", "model_short", "confidence_bin"], observed=False)
        .agg(
            mean_confidence=("confidence_mean", "mean"),
            mean_accuracy=("accuracy", "mean"),
            n=("accuracy", "size"),
        )
        .reset_index()
        .dropna(subset=["mean_confidence", "mean_accuracy"])
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(5, 4),
    )

    model_colors = _model_color_map(data.runs)
    models = [m for m in _sorted_models(ags) if m in model_colors]

    subset = cal[cal["role"] == "Honest"]

    for model in models:
        msub = subset[subset["model_short"] == model]
        if msub.empty:
            continue
        msub = msub.sort_values("mean_confidence")
        ax.plot(
            msub["mean_confidence"],
            msub["mean_accuracy"],
            label=model,
            color=model_colors[model],
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
        ax.scatter(
            msub["mean_confidence"],
            msub["mean_accuracy"],
            color=model_colors[model],
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            s=20 + 2 * msub["n"].clip(upper=40),
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1)
    ax.set_xlabel("Binned mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.text(
        0.05,
        0.95,
        "Honest",
        transform=ax.transAxes,
        fontsize=_FONT_SIZE_SMALL,
        va="top",
        fontweight="bold",
        color=_ROLE_COLORS["Honest"],
    )
    _add_figure_legend(
        fig,
        ax,
        title="Model",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 (replacement) – Expected calibration error table
# ---------------------------------------------------------------------------


def _calibration_bins_for_honest_agents(data: SweepData) -> pd.DataFrame:
    """Return binned calibration stats for honest agents by model."""
    ags = data.agent_summary.dropna(subset=["confidence_mean", "accuracy"]).copy()
    if ags.empty:
        return pd.DataFrame(
            columns=[
                "model_short",
                "confidence_bin",
                "mean_confidence",
                "mean_accuracy",
                "n",
            ]
        )

    ags = ags[ags["role"] == "Honest"].copy()
    if ags.empty:
        return pd.DataFrame(
            columns=[
                "model_short",
                "confidence_bin",
                "mean_confidence",
                "mean_accuracy",
                "n",
            ]
        )

    ags["confidence_mean"] = ags["confidence_mean"].clip(0, 1)
    ags["accuracy"] = ags["accuracy"].clip(0, 1)
    # Match fig_confidence_calibration binning for consistency.
    bin_edges = np.linspace(0, 1, 9)
    ags["confidence_bin"] = pd.cut(
        ags["confidence_mean"],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    return (
        ags.groupby(["model_short", "confidence_bin"], observed=False)
        .agg(
            mean_confidence=("confidence_mean", "mean"),
            mean_accuracy=("accuracy", "mean"),
            n=("accuracy", "size"),
        )
        .reset_index()
        .dropna(subset=["mean_confidence", "mean_accuracy"])
    )


def fig3_ece_table(data: SweepData) -> str:
    """Return a plain-text table of expected calibration error per model."""
    cal = _calibration_bins_for_honest_agents(data)
    if cal.empty:
        return "No calibration data found for honest agents.\n"

    ece = (
        cal.assign(abs_gap=(cal["mean_accuracy"] - cal["mean_confidence"]).abs())
        .groupby("model_short", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "ece": np.average(g["abs_gap"], weights=g["n"]),
                    "samples": int(g["n"].sum()),
                    "bins_used": int((g["n"] > 0).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
        .sort_values(["ece", "model_short"], ascending=[True, True])
    )
    ece["samples"] = ece["samples"].astype(int)
    ece["bins_used"] = ece["bins_used"].astype(int)
    ece["ece"] = ece["ece"].map(lambda x: f"{x:.4f}")

    lines = [
        "# Fig 3 — Expected Calibration Error (Honest Agents)",
        "",
        "Expected calibration error computed over confidence bins (8 equal-width bins on [0, 1]).",
        "",
        ece.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 4 – Liar influence on honest agents
# ---------------------------------------------------------------------------


def fig_liar_influence(data: SweepData) -> plt.Figure:
    """Grouped bar of influence on later agents, by binned liar fraction."""
    am = data.agent_metrics.dropna(subset=["influence_on_later_agents"]).copy()
    am = am[am["num_liars"] > 0]
    if am.empty:
        return _empty_fig("No data with liars found")

    bin_order = [b for b in _LIAR_BIN_ORDER if b in am["liar_share_bin"].values]
    models = _sorted_models(am)
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = am[am["model_short"] == model]
        sns.barplot(
            data=subset,
            x="liar_share_bin",
            order=bin_order,
            y="influence_on_later_agents",
            hue="role",
            hue_order=["Honest", "Liar"],
            palette=_ROLE_COLORS,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(model)
        ax.set_xlabel("Liar fraction" if row == n_rows - 1 else "")
        ax.set_ylabel("Influence on later agents" if col == 0 else "")
        ax.set_ylim(0, 1.05)
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    _add_figure_legend(
        fig,
        visible_axes[0],
        title="Role",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def fig4_alternate(data: SweepData) -> plt.Figure:
    """Liar-minus-honest influence gap by liar-fraction bin."""
    am = data.agent_metrics.dropna(subset=["influence_on_later_agents"]).copy()
    am = am[am["num_liars"] > 0]
    if am.empty:
        return _empty_fig("No data with liars found")

    # Compute a paired gap per run: mean(liar influence) - mean(honest influence).
    role_means = (
        am.groupby(
            ["run_id", "model_short", "liar_share_bin", "role"],
            observed=False,
        )
        .agg(mean_influence=("influence_on_later_agents", "mean"))
        .reset_index()
    )
    gap = role_means.pivot_table(
        index=["run_id", "model_short", "liar_share_bin"],
        columns="role",
        values="mean_influence",
        aggfunc="mean",
    ).reset_index()
    if "Liar" not in gap.columns or "Honest" not in gap.columns:
        return _empty_fig("Cannot compute influence gap: missing liar or honest role")
    gap["influence_gap"] = gap["Liar"] - gap["Honest"]
    gap = gap.dropna(subset=["influence_gap"])
    if gap.empty:
        return _empty_fig("No paired liar/honest influence data found")

    bin_order = [b for b in _LIAR_BIN_ORDER if b in am["liar_share_bin"].values]
    models = _sorted_models(gap)
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = gap[gap["model_short"] == model]
        sns.barplot(
            data=subset,
            x="liar_share_bin",
            order=bin_order,
            y="influence_gap",
            color="#4C72B0",
            errorbar=("ci", 95),
            ax=ax,
        )

        ax.set_title(model)
        ax.set_xlabel("Liar fraction" if row == n_rows - 1 else "")
        ax.set_ylabel("Influence gap (Liar - Honest)" if col == 0 else "")
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1, alpha=0.8)
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    max_abs = float(gap["influence_gap"].abs().max())
    ylim = max(0.1, max_abs * 1.15)
    for ax in visible_axes:
        ax.set_ylim(-ylim, ylim)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 – Honest-agent payoff table (text)
# ---------------------------------------------------------------------------


def fig5_honest_payoff_table(data: SweepData) -> str:
    """Return a plain-text table of honest-agent payoff statistics by model."""

    am = data.agent_metrics.copy()
    am = am[am["role"] == "Honest"]
    if am.empty:
        return "No honest-agent payoff data found.\n"

    summary = (
        am.groupby(["model_short", "liar_share_label"])
        .agg(
            mean_payoff=("realized_payoff", "mean"),
            std=("realized_payoff", "std"),
            n=("realized_payoff", "size"),
        )
        .reset_index()
    )
    if summary.empty:
        return "No honest-agent payoff data found.\n"

    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["n"])
    summary["model"] = summary["model_short"]
    summary["liar_fraction"] = summary["liar_share_label"]
    summary["mean_payoff"] = summary["mean_payoff"].round(3)
    summary["ci95"] = summary["ci95"].round(3)
    summary["mean_payoff_95ci"] = (
        summary["mean_payoff"].map(lambda v: f"{v:.3f}")
        + " +/- "
        + summary["ci95"].map(lambda v: f"{v:.3f}")
    )

    liar_labels = _sorted_liar_labels(am)
    summary["liar_fraction"] = pd.Categorical(
        summary["liar_fraction"],
        categories=liar_labels,
        ordered=True,
    )

    long_table = (
        summary[
            [
                "model",
                "liar_fraction",
                "mean_payoff",
                "ci95",
                "n",
                "mean_payoff_95ci",
            ]
        ]
        .sort_values(["liar_fraction", "model"])
        .rename(
            columns={
                "model": "Model",
                "liar_fraction": "Liar fraction",
                "mean_payoff": "Mean payoff",
                "ci95": "95% CI half-width",
                "n": "Honest-agent samples",
                "mean_payoff_95ci": "Mean payoff (+/- 95% CI)",
            }
        )
    )

    wide_table = summary.pivot(
        index="liar_fraction",
        columns="model",
        values="mean_payoff_95ci",
    )
    wide_table.index.name = "Liar fraction"
    wide_table = wide_table.sort_index()

    lines = [
        "# Fig 5 — Honest-Agent Payoff Table",
        "",
        "Metric: realized payoff for honest agents only.",
        "Reported as mean +/- 95% CI (normal approximation): mean +/- 1.96*SE.",
        "",
        "## Wide View (rows = liar fraction, cols = model)",
        "",
        wide_table.to_string(),
        "",
        "## Detailed View",
        "",
        long_table.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 6 – Parameter heatmap
# ---------------------------------------------------------------------------


def fig_parameter_heatmap(data: SweepData) -> plt.Figure:
    """Heatmap of mean accuracy across num_agents x num_liars.

    Rows = stag_success_threshold, columns = model.
    """
    runs = data.runs
    models = _sorted_models(runs)
    thresholds = sorted(runs["stag_success_threshold"].unique())

    n_rows, n_cols = len(thresholds), len(models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, threshold in enumerate(thresholds):
        for col, model in enumerate(models):
            ax = axes[row, col]
            subset = runs[
                (runs["model_short"] == model)
                & (runs["stag_success_threshold"] == threshold)
            ]
            if subset.empty:
                ax.set_visible(False)
                continue
            pivot = subset.pivot_table(
                values="accuracy",
                index="num_agents",
                columns="num_liars",
                aggfunc="mean",
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar=(col == n_cols - 1),
                cbar_kws={"label": "Mean accuracy"} if col == n_cols - 1 else {},
                ax=ax,
            )
            ax.text(
                0.05,
                1.02,
                f"{model}  (M={threshold})",
                transform=ax.transAxes,
                fontsize=_FONT_SIZE_SMALL,
                va="bottom",
                color="0.4",
            )
            ax.set_xlabel("Number of liars")
            ax.set_ylabel("Number of agents" if col == 0 else "")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7 – Consensus rate dynamics
# ---------------------------------------------------------------------------


def fig_consensus_entropy(data: SweepData) -> plt.Figure:
    """Honest-agent consensus rate over rounds.

    Rows = stag_success_threshold (M)
    Columns = model
    Hue = liar-fraction bin
    """

    rm = _multi_round_filter(data.round_metrics)
    if rm.empty:
        return _empty_fig("No multi-round runs found")
    am = data.agent_metrics.copy()
    am = am[am["run_id"].isin(rm["run_id"])].copy()
    if am.empty:
        return _empty_fig("No agent-level data found for consensus plot")

    # Compute per-round consensus among honest agents only.
    am["reported_is_stag"] = _as_bool(am["reported_is_stag"])
    honest = am[am["role"] == "Honest"].copy()
    if honest.empty:
        return _empty_fig("No honest-agent data found for consensus plot")

    honest_round = honest.groupby(["run_id", "round"], as_index=False).agg(
        num_honest=("reported_is_stag", "size"),
        num_honest_stag=("reported_is_stag", "sum"),
    )
    honest_round["num_honest_hare"] = (
        honest_round["num_honest"] - honest_round["num_honest_stag"]
    )
    honest_round["honest_consensus_rate"] = (
        honest_round[["num_honest_stag", "num_honest_hare"]].max(axis=1)
        / honest_round["num_honest"]
    )
    rm = rm.merge(
        honest_round[["run_id", "round", "honest_consensus_rate"]],
        on=["run_id", "round"],
        how="left",
    )

    thresholds = [
        t for t in sorted(rm["stag_success_threshold"].unique()) if t in (3, 4)
    ]
    if not thresholds:
        return _empty_fig("No data found for M=3 or M=4 in consensus plot")
    rm = rm[rm["stag_success_threshold"].isin(thresholds)].copy()
    models = _sorted_models(rm)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in rm["liar_share_bin"].values]

    n_rows = len(thresholds)
    n_cols = len(models)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
        sharey=True,
    )

    for row, threshold in enumerate(thresholds):
        for col, model in enumerate(models):
            ax = axes[row, col]

            subset = rm[
                (rm["stag_success_threshold"] == threshold)
                & (rm["model_short"] == model)
            ]

            _lineplot_with_errorbars(
                ax=ax,
                data=subset.dropna(subset=["honest_consensus_rate"]),
                x="round",
                y="honest_consensus_rate",
                hue="liar_share_bin",
                hue_order=bin_order,
                palette=_LIAR_SHARE_PALETTE,
                errorbar=("ci", 95),
                dense=True,
            )

            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlabel("Round" if row == n_rows - 1 else "")
            ax.set_ylim(-0.05, 1.05)

            if ax.get_legend():
                ax.get_legend().remove()

    # Column titles = model names
    for col, model in enumerate(models):
        axes[0, col].set_title(model)

    # Row labels embedded in the first-column y-axis label so tight_layout
    # handles spacing automatically (avoids the fragile annotate xy offset).
    for row, threshold in enumerate(thresholds):
        axes[row, 0].set_ylabel(f"M={threshold}\nHonest-agent consensus rate")

    _add_figure_legend(
        fig,
        axes[0, 0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 9 – Coordination dynamics over rounds
# ---------------------------------------------------------------------------


def fig_coordination_over_rounds(data: SweepData) -> plt.Figure:
    """Stag-success rate over rounds, by liar-fraction bin, faceted by model.

    Complements fig 1 (aggregate coordination) by showing how coordination
    evolves across rounds within a game.
    """
    rm = _multi_round_filter(data.round_metrics)
    if rm.empty:
        return _empty_fig("No multi-round runs found")

    models = _sorted_models(rm)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in rm["liar_share_bin"].values]
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = rm[rm["model_short"] == model]
        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="round",
            y="stag_success",
            hue="liar_share_bin",
            hue_order=bin_order,
            palette=_LIAR_SHARE_PALETTE,
            errorbar=("ci", 95),
            dense=True,
        )
        ax.set_title(model)
        ax.set_xlabel("Round" if row == n_rows - 1 else "")
        ax.set_ylabel("Stag success rate" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    _add_figure_legend(
        fig,
        visible_axes[0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 10 – Turn-order effects (A1: fixed speaking order)
# ---------------------------------------------------------------------------


def fig_turn_order_effects(data: SweepData) -> plt.Figure:
    """Accuracy by speaking position for honest agents, by liar fraction.

    Under fixed order (A1), agents always speak at the same turn index.
    Shows whether later-speaking honest agents are more susceptible to
    liar influence (information cascade effect).
    """
    am = data.agent_metrics.copy()
    honest = am[am["role"] == "Honest"]
    if honest.empty:
        return _empty_fig("No honest agent data")

    models = _sorted_models(honest)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in honest["liar_share_bin"].values]
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = honest[honest["model_short"] == model]
        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="turn_index",
            y="is_correct",
            hue="liar_share_bin",
            hue_order=bin_order,
            palette=_LIAR_SHARE_PALETTE,
            errorbar=("ci", 95),
            dense=True,
        )
        ax.set_title(model)
        ax.set_xlabel("Speaking position" if row == n_rows - 1 else "")
        ax.set_ylabel("Honest-agent accuracy" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        # turn_index is 0-based; display as 1-based for readability
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: str(int(x) + 1) if x == int(x) else "")
        )
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    _add_figure_legend(
        fig,
        visible_axes[0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 14-17 – Honest-agent belief-rule analysis
# ---------------------------------------------------------------------------


def fig14_belief_response(data: SweepData) -> plt.Figure:
    """Compare honest-agent action curves under the three belief-rule margins."""
    frames = [
        build_belief_benchmark(data),
        build_carryover_benchmark(data),
        build_trust_weighted_benchmark(data),
    ]
    if any(frame.empty for frame in frames):
        return _empty_fig("No agent-level data found for belief-rule response figure")

    honest = pd.concat(frames, ignore_index=True)
    honest = honest[(honest["role"] == "Honest") & honest["benchmark_defined"]].copy()
    honest = honest.dropna(subset=["q_margin_mid"]).copy()
    if honest.empty:
        return _empty_fig("No eligible honest-agent benchmark rows found")

    counts = (
        honest.groupby(
            ["model_short", "benchmark_rule", "q_margin_mid"], observed=False
        )
        .size()
        .rename("n")
        .reset_index()
    )
    honest = honest.merge(
        counts,
        on=["model_short", "benchmark_rule", "q_margin_mid"],
        how="left",
    )
    honest = honest[honest["n"] >= 8].copy()
    if honest.empty:
        return _empty_fig("No sufficiently populated honest-agent benchmark bins found")

    models = _sorted_models(honest)
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = honest[honest["model_short"] == model].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="q_margin_mid",
            y="original_is_stag",
            hue="benchmark_rule",
            hue_order=_BENCHMARK_RULE_ORDER,
            palette=_BENCHMARK_RULE_COLORS,
            errorbar=("ci", 95),
            dense=True,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(model)
        ax.set_xlabel(
            r"Belief-rule margin $\hat{q}_{\mathrm{rule}} - q^*$"
            if row == n_rows - 1
            else ""
        )
        ax.set_ylabel("Honest P(STAG)" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(
            _BELIEF_MARGIN_BIN_EDGES[0] - 0.05, _BELIEF_MARGIN_BIN_EDGES[-1] + 0.05
        )
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    if not visible_axes:
        return _empty_fig("No eligible honest-agent benchmark rows found")

    _add_figure_legend(
        fig,
        visible_axes[0],
        title="Belief rule",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def fig17_carryover_response(data: SweepData) -> plt.Figure:
    """Honest STAG rate as a function of the carryover-rule margin."""
    return _fig_honest_benchmark_response(
        build_carryover_benchmark(data),
        y_col="original_is_stag",
        ylabel="Honest P(STAG)",
        x_label=r"Carryover-rule margin $\hat{q}_{\mathrm{carry}} - q^*$",
        color=_BENCHMARK_RULE_COLORS["Carryover"],
        empty_message="No agent-level data found for carryover benchmark",
    )


def fig18_trust_response(data: SweepData) -> plt.Figure:
    """Honest STAG rate as a function of the trust-weighted-rule margin."""
    return _fig_honest_benchmark_response(
        build_trust_weighted_benchmark(data),
        y_col="original_is_stag",
        ylabel="Honest P(STAG)",
        x_label=r"Trust-weighted margin $\hat{q}_{\mathrm{trust}} - q^*$",
        color=_BENCHMARK_RULE_COLORS["Trust-weighted"],
        empty_message="No agent-level data found for trust-weighted benchmark",
    )


def fig17_rule_comparison(data: SweepData) -> str:
    """Text summary comparing honest-agent match rates across the three update rules."""
    current = build_belief_benchmark(data)
    carryover = build_carryover_benchmark(data)
    trust = build_trust_weighted_benchmark(data)

    frames = [current, carryover, trust]
    if any(frame.empty for frame in frames):
        return "No agent-level data found for update-rule comparison.\n"

    compare = pd.concat(frames, ignore_index=True)
    compare = compare[
        (compare["role"] == "Honest")
        & compare["benchmark_defined"]
        & (compare["n_observed"] > 0)
    ].copy()
    if compare.empty:
        return "No eligible honest-agent rows found for update-rule comparison.\n"

    compare["benchmark_rule"] = pd.Categorical(
        compare["benchmark_rule"],
        categories=_BENCHMARK_RULE_ORDER,
        ordered=True,
    )
    overall = (
        compare.groupby("benchmark_rule", observed=False)
        .agg(
            Eligible=("matches_benchmark", "size"),
            Match_rate=("matches_benchmark", "mean"),
            Match_se=("matches_benchmark", "sem"),
            Accuracy=("is_correct", "mean"),
            Mean_payoff=("realized_payoff", "mean"),
        )
        .reset_index()
    )

    by_model_stats = (
        compare.groupby(["model_short", "benchmark_rule"], observed=False)
        .agg(
            Match_rate=("matches_benchmark", "mean"),
            Match_se=("matches_benchmark", "sem"),
        )
        .reset_index()
    )
    by_model = by_model_stats.pivot(
        index="model_short", columns="benchmark_rule", values="Match_rate"
    ).reindex(columns=_BENCHMARK_RULE_ORDER)
    by_model_se = by_model_stats.pivot(
        index="model_short", columns="benchmark_rule", values="Match_se"
    ).reindex(columns=_BENCHMARK_RULE_ORDER)
    by_model["Best_rule"] = by_model.idxmax(axis=1)

    overall_fmt = overall.copy()
    for col in ("Match_rate", "Match_se", "Accuracy", "Mean_payoff"):
        overall_fmt[col] = overall_fmt[col].map(lambda x: f"{float(x):.3f}")

    by_model_fmt = pd.DataFrame(index=by_model.index)
    for col in _BENCHMARK_RULE_ORDER:
        by_model_fmt[col] = [
            f"{float(rate):.3f} +/- {float(se):.3f}"
            for rate, se in zip(by_model[col], by_model_se[col], strict=False)
        ]
    by_model_fmt["Best_rule"] = by_model["Best_rule"]

    total_rows = int(
        compare[compare["benchmark_rule"] == _BENCHMARK_RULE_ORDER[0]].shape[0]
    )
    lines = [
        "# Fig 17 — Update-Rule Comparison Table",
        "",
        "Comparison uses honest-agent turns on common support across all three rules.",
        "Eligible rows are turns with at least one observed prior report (n_observed > 0).",
        "Match rate is the share of eligible honest-agent observations where the agent's original action matches the rule-implied rational action.",
        "Accuracy is the share of those same observations where the agent's original action is correct (`is_correct`).",
        "Match-rate standard errors are Bernoulli standard errors computed across eligible honest-agent observations.",
        f"Eligible honest-agent observations per rule: {total_rows}",
        "",
        "Overall summary:",
        overall_fmt.to_string(index=False),
        "",
        "Model-by-model match rate:",
        by_model_fmt.to_string(),
        "",
    ]
    return "\n".join(lines)


def fig15_naive_match_by_turn(data: SweepData) -> plt.Figure:
    """Honest-agent naive-aggregate match rate by speaking position."""
    bench = build_belief_benchmark(data)
    if bench.empty:
        return _empty_fig("No agent-level data found for public-belief benchmark")

    honest = bench[(bench["role"] == "Honest") & bench["benchmark_defined"]].copy()
    if honest.empty:
        return _empty_fig("No eligible honest-agent benchmark rows found")

    models = _sorted_models(honest)
    bin_order = [b for b in _LIAR_BIN_ORDER if b in honest["liar_share_bin"].values]
    n_rows = 2
    n_cols = max(1, math.ceil(len(models) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    flat_axes = list(axes.flat)
    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    visible_axes: list[plt.Axes] = []
    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        subset = honest[honest["model_short"] == model]
        _lineplot_with_errorbars(
            ax=ax,
            data=subset,
            x="turn_index",
            y="matches_benchmark",
            hue="liar_share_bin",
            hue_order=bin_order,
            palette=_LIAR_SHARE_PALETTE,
            errorbar=("ci", 95),
            dense=True,
        )
        ax.set_title(model)
        ax.set_xlabel("Speaking position" if row == n_rows - 1 else "")
        ax.set_ylabel("Benchmark match rate" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: str(int(x) + 1) if x == int(x) else "")
        )
        if ax.get_legend():
            ax.get_legend().remove()
        visible_axes.append(ax)

    _add_figure_legend(
        fig,
        visible_axes[0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def fig16_naive_alignment_table(data: SweepData) -> str:
    """Text summary of honest-agent naive-aggregate alignment by model."""
    bench = build_belief_benchmark(data)
    if bench.empty:
        return "No agent-level data found for public-belief benchmark.\n"

    honest = bench[bench["role"] == "Honest"].copy()
    if honest.empty:
        return "No honest-agent data found for public-belief benchmark.\n"

    rows: list[dict[str, int | float | str]] = []
    for model, grp in honest.groupby("model_short", sort=True):
        honest_obs = int(len(grp))
        eligible = grp[grp["benchmark_defined"]].copy()
        no_prior = honest_obs - int(len(eligible))

        below = eligible[eligible["q_margin"] < 0]
        above = eligible[eligible["q_margin"] > 0]
        near = eligible[eligible["q_margin"].abs() <= 0.1]
        matched = eligible[eligible["matches_benchmark"]]
        mismatched = eligible[~eligible["matches_benchmark"]]

        rows.append(
            {
                "Model": model,
                "Honest obs": honest_obs,
                "Eligible": int(len(eligible)),
                "No-prior": no_prior,
                "Match rate": float(eligible["matches_benchmark"].mean())
                if not eligible.empty
                else float("nan"),
                "False coop": float(below["false_cooperate"].mean())
                if not below.empty
                else float("nan"),
                "False defect": float(above["false_defect"].mean())
                if not above.empty
                else float("nan"),
                "Near-threshold match": float(near["matches_benchmark"].mean())
                if not near.empty
                else float("nan"),
                "Correct if match": float(matched["is_correct"].mean())
                if not matched.empty
                else float("nan"),
                "Correct if mismatch": float(mismatched["is_correct"].mean())
                if not mismatched.empty
                else float("nan"),
                "Payoff if match": float(matched["realized_payoff"].mean())
                if not matched.empty
                else float("nan"),
                "Payoff if mismatch": float(mismatched["realized_payoff"].mean())
                if not mismatched.empty
                else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["Match rate", "Model"], ascending=[False, True]
    ).reset_index(drop=True)

    rate_cols = [
        "Match rate",
        "False coop",
        "False defect",
        "Near-threshold match",
        "Correct if match",
        "Correct if mismatch",
    ]
    payoff_cols = ["Payoff if match", "Payoff if mismatch"]
    for col in rate_cols:
        summary[col] = summary[col].map(
            lambda x: "n/a" if pd.isna(x) else f"{float(x):.3f}"
        )
    for col in payoff_cols:
        summary[col] = summary[col].map(
            lambda x: "n/a" if pd.isna(x) else f"{float(x):.3f}"
        )

    total_honest = int(len(honest))
    total_eligible = int(honest["benchmark_defined"].sum())
    total_no_prior = total_honest - total_eligible

    lines = [
        "# Fig 16 — Naive-Aggregate Alignment Table",
        "",
        "Main comparison uses honest agents only from base/default runs (a1, base, h1).",
        "Eligible rows are turns with an observable prior; no-prior rows are excluded from match-rate statistics.",
        f"Total honest observations: {total_honest}",
        f"Eligible observations: {total_eligible}",
        f"Excluded no-prior observations: {total_no_prior}",
        r"Near-threshold means $|\hat{q} - q^*| \le 0.1$.",
        "",
        summary.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def figure_story_doc(_: SweepData) -> str:
    """Markdown guide describing the purpose/story of each figure."""
    return r"""# Stag Hunt Figure Guide

This document explains what each analysis figure is trying to show, why it exists, and what story to read from it.

## Core coordination and learning figures

### Fig 1 — Coordination Success vs. Liar Fraction
Purpose: Show the main failure mode of the system as adversarial participation increases.

Story: Coordination succeeds reliably at low liar fractions and then degrades as corruption pushes the group below the effective coordination threshold.

### Fig 1 Highlight — N=5, M=3
Purpose: Give one clean focal setting where the coordination transition is easy to inspect.

Story: This is the most presentation-friendly slice of Fig 1 and makes the onset of failure easier to compare across models.

### Fig 2 — Honest-Agent Accuracy & Confidence Over Rounds
Purpose: Compare actual learning quality to self-reported confidence over repeated discussion rounds.

Story: Honest agents can become less accurate even while confidence stays high, which is evidence of confident error under corruption.

### Fig 2 Alternate — Combined Models
Purpose: Put all models on the same figure so cross-model differences are direct.

Story: This emphasizes relative model robustness rather than within-model liar-fraction effects.

### Fig 2 Alternate B — Focused Confidence Gap
Purpose: Highlight the gap between honest confidence and honest accuracy.

Story: This is a compact view of overconfidence dynamics. Positive values mean agents are more confident than their actual accuracy warrants.

### Fig 3 — Expected Calibration Error
Purpose: Summarize whether honest-agent confidence tracks actual correctness.

Story: Lower ECE means confidence is informative; higher ECE means agents are systematically miscalibrated.

## Strategic influence and outcome figures

### Fig 4 — Liar Influence
Purpose: Measure how much earlier speakers shape later speakers' reports.

Story: As liar pressure increases, influence shifts away from honest agents and toward deceptive or corrupted reports.

### Fig 4 Alternate — Liar-Honest Influence Gap
Purpose: Collapse Fig 4 into one signed comparison.

Story: Positive values mean liars are more influential than honest agents in that setting.

### Fig 5 — Honest-Agent Payoffs
Purpose: Connect coordination quality to realized utility for honest players.

Story: As corruption rises, honest agents lose payoff even when they are not the ones lying.

### Fig 6 — Accuracy Heatmap
Purpose: Show the broad shape of performance over the parameter grid.

Story: This is the fastest overview of where the system is robust versus brittle.

### Fig 7 — Consensus Dynamics
Purpose: Track how quickly honest-agent consensus forms or collapses over rounds.

Story: Stable consensus at low liar pressure can give way to fragmentation or corrupted consensus at higher liar pressure.

### Fig 9 — Coordination Dynamics Over Rounds
Purpose: Show whether coordination improves, stalls, or collapses within a game.

Story: This complements Fig 1 by adding temporal structure rather than only aggregate success rates.

### Fig 10 — Turn-Order Effects
Purpose: Identify whether speaking later helps or hurts honest agents.

Story: Later speakers can benefit from more information when reports are trustworthy, but they can also become more exposed to corrupted cascades.

## Ablation tables

### Fig 11 — B3 Matched Table
Purpose: Compare random-noise adversaries against the base adversary under matched settings.

Story: This isolates how much of the effect is specific to deterministic flipping rather than adversarial corruption in general.

### Fig 12 — H1 vs H2 Matched Table
Purpose: Compare homogeneous groups with mixed-model groups under matched parameter points.

Story: This tests whether heterogeneity changes coordination success after controlling for the rest of the setup.

### Fig 13 — H1 vs H3 Matched Table
Purpose: Compare homogeneous groups with asymmetric liar/non-liar model assignment.

Story: This tests whether giving liars a model-strength advantage or disadvantage changes outcomes.

## Public-belief benchmark figures

### Fig 14 — Honest-Agent Action vs Belief-Rule Margin
Purpose: Compare the honest-agent response curves implied by the three benchmark rules on a common set of axes.

Story: The key comparison is whether the carryover and trust-weighted rules produce cleaner threshold-like response curves than the naive aggregate rule.

How Fig 14 changed:
- The first version showed only the naive aggregate rule.
- The current version overlays the naive aggregate, carryover, and trust-weighted response curves inside each model panel.
- This makes the figure a direct visual comparison rather than a single-rule diagnostic.

### Fig 15 — Naive-Aggregate Agreement by Turn
Purpose: Show where in the speaking order honest agents align with the naive aggregate rule.

Story: This identifies whether deviations from the naive aggregate rule are concentrated among early speakers, late speakers, or uniformly across the round.

### Fig 16 — Naive-Aggregate Alignment Table
Purpose: Give a compact model-by-model summary of naive-aggregate agreement and its consequences.

Story: This table shows who matches the naive aggregate rule most often, what kinds of mistakes they make, and whether mismatches are costly in correctness or payoff terms.

### Fig 17 — Update-Rule Comparison Table
Purpose: Compare the naive aggregate rule against the two richer alternatives on a common-support match-rate metric in text-table form.

Story: This is the direct model-selection figure for the benchmark family. Higher values mean the rule better matches honest-agent observed behavior on turns where all three rules are comparable.

## Candidate Alternative Belief Rules

These are candidate extensions to the naive aggregate benchmark. The two strongest variants now appear as overlaid curves in Fig 14, with a direct comparison table in Fig 17.

Notation:
- Let \(y_j^t \in \{0,1\}\) denote agent \(j\)'s public report in round \(t\), where \(1=\mathrm{STAG}\).
- Let \(S_i^t\) be the set of speakers agent \(i\) has observed so far in round \(t\).
- Let \(q_i^t\) denote agent \(i\)'s belief about the public STAG report rate.
- Let \(\hat q_i^t\) denote the rule-implied estimate of that public belief used in the benchmark figures below.
- In this section, beliefs are about the transcript-level / public report process, not a latent true-action rate.

### 1. Carryover-Prior Update

Start each round with a prior anchored in the previous round's posterior:

\[
q_{i,\mathrm{prior}}^{t} = \lambda \hat q_{i,\mathrm{post}}^{t-1} + (1-\lambda) q_0
\]

where \(q_0 \in [0,1]\) is a neutral baseline prior over public STAG reports and \(\lambda \in [0,1]\) controls memory strength.

Update within the round using pseudo-count pooling:

\[
\hat q_i^t = \frac{\tau q_{i,\mathrm{prior}}^{t} + \sum_{j \in S_i^t} y_j^t}{\tau + |S_i^t|}
\]

where \(\tau > 0\) is the prior strength.

Story: agents assume the public reporting environment persists across rounds, but revise that prior as current-round reports arrive.

### 2. Agent-Specific Trust Update

Instead of weighting all speakers equally, let agent \(i\) maintain a trust weight \(\rho_{ij}^t \in [0,1]\) for each other agent \(j\). Here \(\rho_{ij}^t\) should be interpreted as a predictive weight on how informative \(j\)'s public reports are about future public STAG reporting, not as an attempt to recover a hidden true action. Then form a weighted within-round estimate:

\[
\hat q_i^t = \frac{\tau q_{i,\mathrm{prior}}^{t} + \sum_{j \in S_i^t} \rho_{ij}^t y_j^t}{\tau + \sum_{j \in S_i^t} \rho_{ij}^t}
\]

Update trust across rounds using a simple Beta-style reliability score. Let \(z_j^t \in \{0,1\}\) indicate whether agent \(j\)'s public report in round \(t\) matched some target notion of reliability, such as the final round outcome or the final majority report. Maintain

\[
a_{ij}^{t+1} = a_{ij}^{t} + z_j^t, \qquad b_{ij}^{t+1} = b_{ij}^{t} + (1-z_j^t)
\]

and define the trust weight as

\[
\rho_{ij}^{t+1} = \frac{a_{ij}^{t+1}}{a_{ij}^{t+1} + b_{ij}^{t+1}}
\]

Story: agents learn not just how STAG-heavy the public report environment is, but which specific speakers are worth weighting when forecasting future public reports.

### Recommendation for Discussion

If the goal is to present a small, interpretable family of richer benchmarks, the strongest comparison set is:

1. Naive aggregate public-belief rule
2. Carryover-prior update
3. Agent-specific trust update

These span three distinct ideas:
- no memory beyond the current round
- generic memory of the public reporting environment
- memory about individual speaker reliability rather than only aggregate group state
"""


# ---------------------------------------------------------------------------
# Figure 11-13 – Matched ablation tables (text)
# ---------------------------------------------------------------------------


_B3_POINT_KEY_COLS = [
    "model",
    "num_agents",
    "num_rounds",
    "num_liars",
    "stag_success_threshold",
    "payoff_stag_success",
    "payoff_hare_when_stag_success",
    "payoff_stag_fail",
    "payoff_hare_fail",
    "order_ablation",
    "heterogeneity_ablation",
    "h3_liar_policy",
    "model_pool",
    "replicate",
    "seed",
]

_HET_POINT_KEY_COLS = [
    "model",
    "num_agents",
    "num_rounds",
    "num_liars",
    "stag_success_threshold",
    "payoff_stag_success",
    "payoff_hare_when_stag_success",
    "payoff_stag_fail",
    "payoff_hare_fail",
    "order_ablation",
    "adversary_ablation",
    "h3_liar_policy",
    "replicate",
    "seed",
]


def _exact_mcnemar_pvalue(n_01: int, n_10: int) -> float:
    """Exact two-sided McNemar p-value via Binomial(n_01+n_10, 0.5)."""
    n_disc = n_01 + n_10
    if n_disc == 0:
        return 1.0

    k = min(n_01, n_10)
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2**n_disc)
    return min(1.0, 2 * tail)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _matched_variant_round_table(
    data: SweepData,
    *,
    heading: str,
    baseline_label: str,
    variant_label: str,
    variant_points: pd.DataFrame,
    variant_runs: pd.DataFrame,
    point_key_cols: list[str],
) -> str:
    """Build a matched baseline-vs-variant roundwise significance table."""
    runs = data.runs.copy()
    rm = data.round_metrics.copy()
    points = data.sweep_points.copy()

    if runs.empty or rm.empty or points.empty:
        return f"No run/round data found for {variant_label} matched analysis.\n"

    required_point_cols = {"point_id", "ablation_code", *point_key_cols}
    if not required_point_cols.issubset(points.columns):
        return (
            "Sweep-point metadata is missing required columns for deterministic "
            f"{variant_label}/{baseline_label} pairing.\n"
        )

    runs["point_id"] = runs["run_id"].astype(str).str.rsplit("_", n=1).str[-1]
    if "point_id" not in variant_runs.columns and "run_id" in variant_runs.columns:
        variant_runs = variant_runs.copy()
        variant_runs["point_id"] = (
            variant_runs["run_id"].astype(str).str.rsplit("_", n=1).str[-1]
        )
    if "model_pool" in points.columns:
        points["model_pool"] = points["model_pool"].fillna("").astype(str)
    if "model_pool" in variant_points.columns:
        variant_points["model_pool"] = (
            variant_points["model_pool"].fillna("").astype(str)
        )

    base_points = points[points["ablation_code"] == "base"].copy()
    if base_points.empty:
        return "No base sweep points found for matched ablation analysis.\n"
    if variant_points.empty:
        return f"No {variant_label} ablation sweep points found.\n"
    if variant_runs.empty:
        return f"No {variant_label} ablation runs found.\n"

    base_points_keyed = (
        base_points.groupby(point_key_cols, as_index=False)
        .agg(base_point_id=("point_id", "first"), n_base_points=("point_id", "size"))
        .copy()
    )
    variant_to_base_points = (
        variant_points.merge(
            base_points_keyed,
            on=point_key_cols,
            how="left",
        )
        .rename(columns={"point_id": "variant_point_id"})
        .drop_duplicates(subset=["variant_point_id"])
    )

    base_ids = set(base_points["point_id"])
    base_run_lookup = (
        runs[runs["point_id"].isin(base_ids)]
        .groupby("point_id", as_index=False)
        .agg(base_run_id=("run_id", "first"), n_base_run_candidates=("run_id", "size"))
        .rename(columns={"point_id": "base_point_id"})
    )
    if base_run_lookup.empty:
        return "No base runs found to match against variant runs.\n"

    pairs = (
        variant_runs.merge(
            variant_to_base_points[
                ["variant_point_id", "base_point_id", "n_base_points"]
            ],
            left_on="point_id",
            right_on="variant_point_id",
            how="left",
        )
        .merge(base_run_lookup, on="base_point_id", how="left")
        .copy()
    )

    total_variant = int(len(variant_runs))
    unmatched = int(pairs["base_point_id"].isna().sum())
    ambiguous_points = int((pairs["n_base_points"].fillna(0) > 1).sum())
    missing_base_runs = int(
        (pairs["base_point_id"].notna() & pairs["base_run_id"].isna()).sum()
    )
    ambiguous_base_runs = int((pairs["n_base_run_candidates"].fillna(0) > 1).sum())

    valid_pairs = pairs[
        pairs["base_run_id"].notna()
        & (pairs["n_base_points"] == 1)
        & (pairs["n_base_run_candidates"] == 1)
    ][["run_id", "base_run_id"]].rename(columns={"run_id": "variant_run_id"})
    valid_pairs = valid_pairs.drop_duplicates()

    if valid_pairs.empty:
        lines = [
            heading,
            "",
            f"No unambiguous {variant_label}/{baseline_label} matched pairs were found.",
            f"Total {variant_label} runs: {total_variant}",
            f"Unmatched {variant_label} runs: {unmatched}",
            f"Ambiguous point matches (>1 base point): {ambiguous_points}",
            f"Missing base runs for mapped points: {missing_base_runs}",
            f"Ambiguous base runs (>1 run for base point): {ambiguous_base_runs}",
            "",
        ]
        return "\n".join(lines)

    rm = rm[["run_id", "round", "stag_success"]].copy()
    rm["stag_success"] = _as_bool(rm["stag_success"])

    base_round = rm.rename(
        columns={"run_id": "base_run_id", "stag_success": "base_stag_success"}
    )
    variant_round = rm.rename(
        columns={"run_id": "variant_run_id", "stag_success": "variant_stag_success"}
    )

    paired_rounds = (
        valid_pairs.merge(base_round, on="base_run_id", how="inner")
        .merge(variant_round, on=["variant_run_id", "round"], how="inner")
        .copy()
    )
    if paired_rounds.empty:
        lines = [
            heading,
            "",
            "Matched run pairs found, but no overlapping round-level observations.",
            f"Matched run pairs: {len(valid_pairs)}",
            "",
        ]
        return "\n".join(lines)

    rows: list[dict[str, float | int | str]] = []
    for round_num, grp in paired_rounds.groupby("round", sort=True):
        n_pairs = int(len(grp))
        base_rate = float(grp["base_stag_success"].mean())
        variant_rate = float(grp["variant_stag_success"].mean())

        n_01 = int(
            ((~grp["base_stag_success"]) & grp["variant_stag_success"]).sum()
        )  # base=0,variant=1
        n_10 = int(
            (grp["base_stag_success"] & (~grp["variant_stag_success"])).sum()
        )  # base=1,variant=0
        n_disc = n_01 + n_10
        p_value = _exact_mcnemar_pvalue(n_01, n_10)

        rows.append(
            {
                "Round": int(round_num),
                "Matched pairs": n_pairs,
                f"{baseline_label} rate": base_rate,
                f"{variant_label} rate": variant_rate,
                "Delta (pp)": 100 * (variant_rate - base_rate),
                "0->1 pairs": n_01,
                "1->0 pairs": n_10,
                "Discordant": n_disc,
                "McNemar p": p_value,
            }
        )

    summary = pd.DataFrame(rows).sort_values("Round").reset_index(drop=True)
    summary["Holm p"] = _holm_adjust(summary["McNemar p"].tolist())
    summary["Significant (Holm<0.05)"] = summary["Holm p"] < 0.05

    summary[f"{baseline_label} rate"] = summary[f"{baseline_label} rate"].map(
        lambda x: f"{x:.3f}"
    )
    summary[f"{variant_label} rate"] = summary[f"{variant_label} rate"].map(
        lambda x: f"{x:.3f}"
    )
    summary["Delta (pp)"] = summary["Delta (pp)"].map(lambda x: f"{x:+.2f}")
    summary["McNemar p"] = summary["McNemar p"].map(lambda x: f"{x:.4g}")
    summary["Holm p"] = summary["Holm p"].map(lambda x: f"{x:.4g}")

    lines = [
        heading,
        "",
        f"Rows are rounds; each row compares matched {variant_label}/{baseline_label} run pairs at that round.",
        "Statistical test: exact McNemar (paired binary outcomes) with Holm correction across rounds.",
        f"0->1 pairs means {baseline_label}=0 and {variant_label}=1.",
        "",
        f"Total {variant_label} runs: {total_variant}",
        f"Matched pairs used: {len(valid_pairs)}",
        f"Unmatched {variant_label} runs: {unmatched}",
        f"Ambiguous point matches (>1 base point): {ambiguous_points}",
        f"Missing base runs for mapped points: {missing_base_runs}",
        f"Ambiguous base runs (>1 run for base point): {ambiguous_base_runs}",
        "",
        summary.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def fig11_b3_matched_round_table(data: SweepData) -> str:
    """Matched b3-vs-base stag-success comparison by round (text table)."""
    points = data.sweep_points.copy()
    runs = data.runs.copy()

    if points.empty:
        return "No sweep-point metadata found for b3 matched analysis.\n"
    if runs.empty or "adversary_ablation" not in runs.columns:
        return "No adversary_ablation run metadata found for b3 matched analysis.\n"

    b3_mask = points["ablation_code"] == "b3"
    if "adversary_ablation" in points.columns:
        b3_mask = b3_mask | (points["adversary_ablation"] == "b3")
    b3_points = points[b3_mask].copy()
    b3_runs = runs[runs["adversary_ablation"] == "b3"].copy()

    return _matched_variant_round_table(
        data,
        heading="# Fig 11 — B3 Matched Roundwise Coordination (Text Table)",
        baseline_label="Base",
        variant_label="B3",
        variant_points=b3_points,
        variant_runs=b3_runs,
        point_key_cols=_B3_POINT_KEY_COLS,
    )


def fig12_h1_vs_h2_matched_table(data: SweepData) -> str:
    """Matched h2-vs-h1 stag-success comparison by round (text table)."""
    points = data.sweep_points.copy()
    runs = data.runs.copy()

    if points.empty:
        return "No sweep-point metadata found for h2 matched analysis.\n"
    if runs.empty or "heterogeneity_ablation" not in runs.columns:
        return "No heterogeneity_ablation run metadata found for h2 matched analysis.\n"

    h2_mask = points["ablation_code"] == "h2"
    if "heterogeneity_ablation" in points.columns:
        h2_mask = h2_mask | (points["heterogeneity_ablation"] == "h2")
    h2_points = points[h2_mask].copy()
    h2_runs = runs[runs["heterogeneity_ablation"] == "h2"].copy()

    return _matched_variant_round_table(
        data,
        heading="# Fig 12 — H1 vs H2 Matched Roundwise Coordination (Text Table)",
        baseline_label="H1",
        variant_label="H2",
        variant_points=h2_points,
        variant_runs=h2_runs,
        point_key_cols=_HET_POINT_KEY_COLS,
    )


def fig13_h1_vs_h3_matched_table(data: SweepData) -> str:
    """Matched h3-vs-h1 stag-success comparison by round (text table)."""
    points = data.sweep_points.copy()
    runs = data.runs.copy()

    if points.empty:
        return "No sweep-point metadata found for h3 matched analysis.\n"
    if runs.empty or "heterogeneity_ablation" not in runs.columns:
        return "No heterogeneity_ablation run metadata found for h3 matched analysis.\n"

    h3_mask = points["ablation_code"] == "h3"
    if "heterogeneity_ablation" in points.columns:
        h3_mask = h3_mask | (points["heterogeneity_ablation"] == "h3")
    h3_points = points[h3_mask].copy()
    h3_runs = runs[runs["heterogeneity_ablation"] == "h3"].copy()

    return _matched_variant_round_table(
        data,
        heading="# Fig 13 — H1 vs H3 Matched Roundwise Coordination (Text Table)",
        baseline_label="H1",
        variant_label="H3",
        variant_points=h3_points,
        variant_runs=h3_runs,
        point_key_cols=_HET_POINT_KEY_COLS,
    )


# ---------------------------------------------------------------------------
# Figure registry & generation
# ---------------------------------------------------------------------------

FIGURE_REGISTRY: dict[str, tuple[str, callable]] = {
    "fig1_coordination": (
        "Coordination Success vs. Liar Fraction",
        fig_coordination_vs_liar_share,
    ),
    "fig1_highlight": (
        "Figure 1 Highlight (N=5, M=3)",
        fig1_highlight,
    ),
    "fig2_accuracy_confidence": (
        "Honest-Agent Accuracy & Confidence Over Rounds",
        fig_accuracy_over_rounds,
    ),
    "fig2_alternate": (
        "Figure 2 Alternate (Combined Models)",
        fig2_alternate,
    ),
    "fig2_alternate_b": (
        "Figure 2 Alternate B (Focused Confidence Gap)",
        fig2_alternate_b,
    ),
    "fig3_calibration": (
        "Expected Calibration Error",
        fig3_ece_table,
    ),
    "fig4_influence": (
        "Liar Influence (Binned Liar Fraction)",
        fig_liar_influence,
    ),
    "fig4_alternate": (
        "Liar-Honest Influence Gap",
        fig4_alternate,
    ),
    "fig5_payoffs": (
        "Honest-agent payoff table",
        fig5_honest_payoff_table,
    ),
    "fig6_heatmap": (
        "Accuracy Heatmap by Threshold",
        fig_parameter_heatmap,
    ),
    "fig7_consensus_entropy": (
        "Consensus Dynamics",
        fig_consensus_entropy,
    ),
    "fig9_coordination_dynamics": (
        "Coordination Dynamics Over Rounds",
        fig_coordination_over_rounds,
    ),
    "fig10_turn_order": (
        "Turn-Order Effects (A1: Fixed Order)",
        fig_turn_order_effects,
    ),
    "fig14_belief_response": (
        "Honest-Agent Action vs Belief-Rule Margin",
        fig14_belief_response,
    ),
    "fig15_naive_match_by_turn": (
        "Naive-Aggregate Agreement by Turn",
        fig15_naive_match_by_turn,
    ),
    "fig16_naive_alignment_table": (
        "Naive-Aggregate Alignment Table",
        fig16_naive_alignment_table,
    ),
    "fig17_rule_comparison": (
        "Update-Rule Comparison Table",
        fig17_rule_comparison,
    ),
    "figure_story_doc": (
        "Figure Purpose and Story Guide",
        figure_story_doc,
    ),
    "fig11_b3_matched_table": (
        "B3 Ablation Matched Roundwise Table",
        fig11_b3_matched_round_table,
    ),
    "fig12_h1_vs_h2_table": (
        "H1 vs H2 Matched Roundwise Table",
        fig12_h1_vs_h2_matched_table,
    ),
    "fig13_h1_vs_h3_table": (
        "H1 vs H3 Matched Roundwise Table",
        fig13_h1_vs_h3_matched_table,
    ),
}

_NON_ABLATION_FIGURE_KEYS = {
    "fig1_coordination",
    "fig1_highlight",
    "fig2_accuracy_confidence",
    "fig2_alternate",
    "fig2_alternate_b",
    "fig3_calibration",
    "fig4_influence",
    "fig4_alternate",
    "fig5_payoffs",
    "fig6_heatmap",
    "fig7_consensus_entropy",
    "fig9_coordination_dynamics",
    "fig10_turn_order",
    "fig14_belief_response",
    "fig15_naive_match_by_turn",
    "fig16_naive_alignment_table",
    "fig17_rule_comparison",
    "figure_story_doc",
}


def generate_all_figures(
    logs_dir: str | Path,
    output_dir: str | Path,
    *,
    fmt: str = "png",
    figure_keys: list[str] | None = None,
) -> list[Path]:
    """Load data, generate figures, and save them to *output_dir*.

    Parameters
    ----------
    logs_dir:
        Directory containing sweep CSV files.
    output_dir:
        Where to write the generated image files.
    fmt:
        Image format (``'png'``, ``'pdf'``, ``'svg'``).
    figure_keys:
        Subset of ``FIGURE_REGISTRY`` keys to generate.  *None* → all.

    Returns
    -------
    list[Path]
        Paths to the saved figure files.
    """
    _apply_style()
    data = load_sweep_data(logs_dir)
    base_data = _base_only_data(data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = figure_keys or list(FIGURE_REGISTRY.keys())
    saved: list[Path] = []
    for key in keys:
        title, fn = FIGURE_REGISTRY[key]
        print(f"  generating {key}: {title} ...")
        figure_data = base_data if key in _NON_ABLATION_FIGURE_KEYS else data
        result = fn(figure_data)
        # If function returns a matplotlib Figure → save it
        if isinstance(result, plt.Figure):
            path = out / f"{key}.{fmt}"
            result.savefig(path)
            plt.close(result)
            saved.append(path)
            print(f"    -> {path}")
        elif isinstance(result, str):
            suffix = ".md" if key == "figure_story_doc" else ".txt"
            path = out / f"{key}{suffix}"
            path.write_text(result, encoding="utf-8")
            saved.append(path)
            print(f"    -> {path}")
        else:
            # Table or non-figure output
            print(f"    (no figure generated for {key})")
    return saved


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis figures for Stag Hunt simulation results.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing sweep CSV files (default: logs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated figures (default: output/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(FIGURE_REGISTRY.keys()),
        default=None,
        help="Generate only specific figures (default: all)",
    )
    args = parser.parse_args()

    saved = generate_all_figures(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        fmt=args.format,
        figure_keys=args.figures,
    )
    print(f"\nDone. {len(saved)} figure(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
