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

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_THEME = "whitegrid"
_MODEL_PALETTE = "Set2"
_ROLE_COLORS = {"Honest": "#4C72B0", "Liar": "#DD8452"}
_LIAR_SHARE_PALETTE = "viridis_r"
_FIG_SINGLE = (8, 5)
_FIG_WIDE = (12, 5)

_LIAR_BIN_ORDER = ["0\u201325%", "25\u201350%", "50\u201375%", "75\u2013100%"]


def _short_model(name: str) -> str:
    """``'openai/gpt-5-mini'`` -> ``'gpt-5-mini'``."""
    return name.rsplit("/", 1)[-1] if "/" in name else name


def _apply_style() -> None:
    """Apply a consistent visual theme to all figures."""
    sns.set_theme(style=_THEME)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
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
) -> None:
    """Consistent lineplot styling with error bars instead of filled bands."""
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        marker="o",
        linewidth=1.8,
        palette=palette,
        errorbar=errorbar,
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 1},
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

    models = _sorted_models(run_agg)
    # Drop M=1 threshold rows — success is trivially near-1.0 when only
    # one agent needs to choose stag, making the row uninformative.
    all_thresholds = sorted(run_agg["stag_success_threshold"].unique())
    thresholds = all_thresholds  # fallback: keep everything

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
            palette=_MODEL_PALETTE,
            errorbar=("ci", 95),
            sort=False,
        )

        # Add vertical cutoff line at m/n
        cutoff = threshold / n_agents
        ax.axvline(
            x=cutoff,
            linestyle="--",
            color="black",
            alpha=0.6,
        )
        row, col = divmod(idx, n_cols)
        ax.set_title(f"N={n_agents}, M={threshold}", fontsize=10)
        ax.set_xlabel("Liar fraction" if row == n_rows - 1 else "")
        ax.set_ylabel("Stag success rate" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=0)
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
            )
            ax.text(
                0.05,
                0.05,
                model,
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="bottom",
                color="0.5",
            )
            ax.set_xlabel("Round" if row == n_rows - 1 else "")
            ax.set_ylabel(metric_label if col == 0 else "")
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if ax.get_legend():
                ax.get_legend().remove()

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

    models = _sorted_models(ags)
    palette = sns.color_palette(_MODEL_PALETTE, n_colors=len(models))
    model_colors = dict(zip(models, palette))

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
        fontsize=9,
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

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=_FIG_WIDE,
        sharey=True,
        squeeze=False,
    )

    for col, model in enumerate(models):
        ax = axes[0, col]
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
        ax.text(
            0.05,
            0.95,
            model,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            color="0.4",
        )
        ax.set_xlabel("Liar fraction")
        ax.set_ylabel("Influence on later agents" if col == 0 else "")
        ax.set_ylim(0, 1.05)
        if ax.get_legend():
            ax.get_legend().remove()

    _add_figure_legend(
        fig,
        axes[0, 0],
        title="Role",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )
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
                fontsize=8,
                va="bottom",
                color="0.4",
            )
            ax.set_xlabel("Number of liars")
            ax.set_ylabel("Number of agents" if col == 0 else "")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7 – Consensus rate and report entropy dynamics
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

    thresholds = sorted(rm["stag_success_threshold"].unique())
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

            sns.lineplot(
                data=subset.dropna(subset=["honest_consensus_rate"]),
                x="round",
                y="honest_consensus_rate",
                hue="liar_share_bin",
                hue_order=bin_order,
                marker="o",
                palette=_LIAR_SHARE_PALETTE,
                errorbar=("ci", 95),
                ax=ax,
            )

            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlabel("Round" if row == n_rows - 1 else "")
            ax.set_ylabel("Honest-agent consensus rate" if col == 0 else "")
            ax.set_ylim(-0.05, 1.05)

            if ax.get_legend():
                ax.get_legend().remove()

    # Column titles = model names
    for col, model in enumerate(models):
        axes[0, col].set_title(model)

    # Row labels = threshold
    for row, threshold in enumerate(thresholds):
        axes[row, 0].annotate(
            f"M={threshold}",
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            fontsize=12,
        )

    _add_figure_legend(
        fig,
        axes[0, 0],
        title="Liar fraction",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
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

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=_FIG_WIDE,
        sharey=True,
        squeeze=False,
    )

    for col, model in enumerate(models):
        ax = axes[0, col]
        subset = rm[rm["model_short"] == model]
        sns.lineplot(
            data=subset,
            x="round",
            y="stag_success",
            hue="liar_share_bin",
            hue_order=bin_order,
            marker="o",
            palette=_LIAR_SHARE_PALETTE,
            errorbar=("ci", 68),
            ax=ax,
        )
        ax.text(
            0.05,
            0.05,
            model,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            color="0.4",
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Stag success rate" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ax.get_legend():
            ax.get_legend().remove()

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

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=_FIG_WIDE,
        sharey=True,
        squeeze=False,
    )

    for col, model in enumerate(models):
        ax = axes[0, col]
        subset = honest[honest["model_short"] == model]
        sns.lineplot(
            data=subset,
            x="turn_index",
            y="is_correct",
            hue="liar_share_bin",
            hue_order=bin_order,
            marker="o",
            palette=_LIAR_SHARE_PALETTE,
            errorbar=("ci", 68),
            ax=ax,
        )
        ax.text(
            0.05,
            0.05,
            model,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            color="0.4",
        )
        ax.set_xlabel("Speaking position")
        ax.set_ylabel("Honest-agent accuracy" if col == 0 else "")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ax.get_legend():
            ax.get_legend().remove()

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
        variant_points["model_pool"] = variant_points["model_pool"].fillna("").astype(
            str
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
    "fig2_accuracy_confidence": (
        "Honest-Agent Accuracy & Confidence Over Rounds",
        fig_accuracy_over_rounds,
    ),
    "fig3_calibration": (
        "Expected Calibration Error (table)",
        fig3_ece_table,
    ),
    "fig4_influence": (
        "Liar Influence (Binned Liar Fraction)",
        fig_liar_influence,
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
        "Consensus & Entropy Dynamics",
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
    "fig2_accuracy_confidence",
    "fig3_calibration",
    "fig4_influence",
    "fig5_payoffs",
    "fig6_heatmap",
    "fig7_consensus_entropy",
    "fig9_coordination_dynamics",
    "fig10_turn_order",
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
            path = out / f"{key}.txt"
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
