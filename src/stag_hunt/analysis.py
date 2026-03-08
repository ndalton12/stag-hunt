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


def _find_csvs(logs_dir: Path, suffix: str) -> list[Path]:
    """Return all CSVs in *logs_dir* whose name ends with *suffix*."""
    return sorted(logs_dir.glob(f"*{suffix}"))


def _read_and_concat(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate multiple CSVs, dropping exact duplicates."""
    if not paths:
        raise FileNotFoundError("No CSV files found for pattern")
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

    # Enrich agent and round tables with run-level parameters and ablation codes
    run_params = runs[
        [
            "run_id",
            "num_agents",
            "num_liars",
            "num_rounds",
            "stag_success_threshold",
            "order_ablation",
            "adversary_ablation",
            "heterogeneity_ablation",
        ]
    ].drop_duplicates()
    agent_summary = agent_summary.merge(run_params, on="run_id", how="left")
    agent_metrics = agent_metrics.merge(run_params, on="run_id", how="left")
    _ablation_cols = ["order_ablation", "adversary_ablation", "heterogeneity_ablation"]
    _merge_cols = ["run_id"] + [
        c for c in _ablation_cols if c not in round_metrics.columns
    ]
    if len(_merge_cols) > 1:
        round_metrics = round_metrics.merge(
            run_params[_merge_cols].drop_duplicates(),
            on="run_id",
            how="left",
        )

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
            columns=["model_short", "confidence_bin", "mean_confidence", "mean_accuracy", "n"]
        )

    ags = ags[ags["role"] == "Honest"].copy()
    if ags.empty:
        return pd.DataFrame(
            columns=["model_short", "confidence_bin", "mean_confidence", "mean_accuracy", "n"]
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

    honest_round = (
        honest.groupby(["run_id", "round"], as_index=False)
        .agg(
            num_honest=("reported_is_stag", "size"),
            num_honest_stag=("reported_is_stag", "sum"),
        )
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
# Shared ablation figure helper
# ---------------------------------------------------------------------------

_ORDER_ABLATION_LABELS = {"a1": "A1: fixed", "a2": "A2: random", "a3": "A3: reversed"}
_ADVERSARY_ABLATION_LABELS = {"base": "Base: flip", "b3": "B3: random noise"}
_HETEROGENEITY_ABLATION_LABELS = {
    "h1": "H1: homogeneous",
    "h2": "H2: mixed",
    "h3": "H3: adversarial",
}
_ABLATION_PALETTE = "Set1"


_PARAM_POINT_COLS = [
    "model",
    "num_agents",
    "num_liars",
    "num_rounds",
    "stag_success_threshold",
]


def _ablation_coordination_figure(
    data: SweepData,
    ablation_col: str,
    ablation_labels: dict[str, str],
    title_suffix: str,
) -> plt.Figure:
    """Bar + strip plot comparing stag-success rate across ablation variants.

    Shared implementation for order / adversary / heterogeneity ablation figures.

    Because ablation runs typically cover a small subset of parameter points,
    we restrict to parameter points that have data for *every* ablation
    variant, ensuring an apples-to-apples comparison.  Each dot is one run's
    mean stag-success rate; bars show the overall mean with 95% CI.
    """
    rm = data.round_metrics.copy()
    if ablation_col not in rm.columns:
        return _empty_fig(f"No {ablation_col} column found")

    rm = rm.dropna(subset=[ablation_col])
    variants = sorted(rm[ablation_col].unique())
    if len(variants) < 2:
        return _empty_fig(f"Only one {ablation_col} value found — nothing to compare")

    rm["_ablation_label"] = (
        rm[ablation_col].map(ablation_labels).fillna(rm[ablation_col])
    )

    label_order = [ablation_labels.get(v, v) for v in variants]

    # Aggregate per run
    group_cols = ["run_id", "_ablation_label", "liar_share"] + [
        c for c in _PARAM_POINT_COLS if c in rm.columns
    ]
    run_agg = (
        rm.groupby(group_cols)
        .agg(stag_success_rate=("stag_success", "mean"))
        .reset_index()
    )

    # Build a parameter-point key and keep only points present in ALL variants
    pp_cols = [c for c in _PARAM_POINT_COLS if c in run_agg.columns]
    run_agg["_pp_key"] = run_agg[pp_cols].astype(str).agg("|".join, axis=1)
    keys_per_variant = run_agg.groupby("_ablation_label")["_pp_key"].apply(set)
    common_keys = set.intersection(*keys_per_variant)
    if not common_keys:
        return _empty_fig(f"No shared parameter points across {ablation_col} variants")

    run_agg = run_agg[run_agg["_pp_key"].isin(common_keys)]
    run_agg["liar_share_rounded"] = run_agg["liar_share"].round(2)

    allowed = [0.0, 0.2, 0.4, 0.6, 0.8]
    run_agg = run_agg[run_agg["liar_share_rounded"].isin(allowed)]

    liar_order = sorted(run_agg["liar_share_rounded"].unique())

    g = sns.catplot(
        data=run_agg,
        x="_ablation_label",
        y="stag_success_rate",
        col="liar_share_rounded",
        col_order=liar_order,
        col_wrap=3,
        kind="bar",
        order=label_order,
        errorbar=("ci", 95),
        sharey=True,
        height=4,
        aspect=0.9,
        palette=_ABLATION_PALETTE,
    )

    g.set_axis_labels("", "Stag success rate")
    g.set_titles("Liar fraction = {col_name}")
    g.set(ylim=(-0.05, 1.05))

    for ax, liar_value in zip(g.axes.flat, liar_order):
        subset = run_agg[run_agg["liar_share_rounded"] == liar_value]

        sns.stripplot(
            data=subset,
            x="_ablation_label",
            y="stag_success_rate",
            color="black",
            alpha=0.25,
            size=3,
            jitter=0.2,
            ax=ax,
        )

        ax.tick_params(axis="x", rotation=20)

    g.fig.tight_layout()

    g.fig.suptitle(
        f"Coordination vs. {title_suffix} across liar fractions",
        y=1.05,
        fontsize=14,
    )

    return g.fig


# ---------------------------------------------------------------------------
# Figure 12 – Order ablation comparison (a1 vs a2 vs a3)
# ---------------------------------------------------------------------------


def fig_order_ablation(data: SweepData) -> plt.Figure:
    """Coordination rate vs liar fraction, comparing speaking-order variants."""
    return _ablation_coordination_figure(
        data,
        "order_ablation",
        _ORDER_ABLATION_LABELS,
        "Speaking order",
    )


# ---------------------------------------------------------------------------
# Figure 13 – Adversary ablation (base vs b3)
# ---------------------------------------------------------------------------


def fig_adversary_ablation(data: SweepData) -> plt.Figure:
    """Coordination rate vs liar fraction, comparing adversary strategies."""
    return _ablation_coordination_figure(
        data,
        "adversary_ablation",
        _ADVERSARY_ABLATION_LABELS,
        "Adversary type",
    )


# ---------------------------------------------------------------------------
# Figure 14 – Heterogeneity ablation (h1 vs h2 vs h3)
# ---------------------------------------------------------------------------


def fig_heterogeneity_ablation(data: SweepData) -> plt.Figure:
    """Coordination rate vs liar fraction, comparing model-composition strategies."""
    return _ablation_coordination_figure(
        data,
        "heterogeneity_ablation",
        _HETEROGENEITY_ABLATION_LABELS,
        "Model composition",
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
    "fig12_order_ablation": (
        "Order Ablation (A1 vs A2 vs A3)",
        fig_order_ablation,
    ),
    "fig13_adversary_ablation": (
        "Adversary Ablation (Base vs B3)",
        fig_adversary_ablation,
    ),
    "fig14_heterogeneity_ablation": (
        "Heterogeneity Ablation (H1 vs H2 vs H3)",
        fig_heterogeneity_ablation,
    ),
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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = figure_keys or list(FIGURE_REGISTRY.keys())
    saved: list[Path] = []
    for key in keys:
        title, fn = FIGURE_REGISTRY[key]
        print(f"  generating {key}: {title} ...")
        result = fn(data)
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
