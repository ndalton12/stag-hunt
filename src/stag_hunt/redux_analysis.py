"""Mechanism-focused post-hoc analysis for the Stag Hunt experiments.

This module leaves the original analysis suite untouched and writes a separate
``<logs-name>_redux`` output tree.  It uses only existing CSV logs: no model
calls or new simulations are made.

Usage::

    uv run python -m stag_hunt.redux_analysis \
        --logs-dir logs/all_combination
"""

from __future__ import annotations

import argparse
import math
import numbers
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from stag_hunt.analysis import (
    SweepData,
    _apply_style as _apply_base_style,
    build_belief_benchmark,
    build_carryover_benchmark,
    build_trust_weighted_benchmark,
    load_sweep_data,
)
from stag_hunt.beliefs import compute_q_star
from stag_hunt.sim import AGENT_SYSTEM_PROMPT


_FOCAL_N = 5
_FOCAL_M = 3
_BOOTSTRAP_REPLICATES = 1_000
_BOOTSTRAP_SEED = 20260720

_FONT_SIZE_BASE = 23
_FONT_SIZE_AXES = 24
_FONT_SIZE_TICKS = 19
_FONT_SIZE_FACET = 22
_FONT_SIZE_LEGEND = 18
_SAVE_PAD_INCHES = 0.25

_INTENDED_COLOR = "#2A9D8F"
_PUBLIC_COLOR = "#C44E52"
_HONEST_COLOR = "#4C72B0"
_ORIGINAL_HISTORY_COLOR = "#8172B2"

_CORRUPTION_STYLES = {
    0.2: ("#55A868", "o", "-"),
    0.4: ("#2A9D8F", "s", "--"),
    0.6: ("#4C72B0", "^", "-."),
    0.8: ("#8172B2", "D", ":"),
}

_PUBLIC_MARGIN_BIN_EDGES = np.arange(-0.75, 1.051, 0.10)

_BENCHMARK_RULE_STYLES = {
    "Naive aggregate": (_HONEST_COLOR, "o", "-"),
    "Carryover": (_INTENDED_COLOR, "s", "--"),
    "Trust-weighted": (_ORIGINAL_HISTORY_COLOR, "^", "-."),
}

_RUN_METADATA_COLUMNS = [
    "model",
    "seed",
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
    "heterogeneity_ablation",
]

_CELL_COLUMNS = [
    "model_short",
    "num_agents",
    "stag_success_threshold",
    "num_liars",
    "num_rounds",
    "seed",
]


def _apply_redux_style() -> None:
    """Use typography that remains legible after paper-width downscaling."""
    _apply_base_style()
    plt.rcParams.update(
        {
            "font.size": _FONT_SIZE_BASE,
            "axes.titlesize": _FONT_SIZE_FACET,
            "axes.labelsize": _FONT_SIZE_AXES,
            "xtick.labelsize": _FONT_SIZE_TICKS,
            "ytick.labelsize": _FONT_SIZE_TICKS,
            "legend.fontsize": _FONT_SIZE_LEGEND,
            "legend.title_fontsize": _FONT_SIZE_LEGEND,
        }
    )


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce bool-like CSV values without treating non-empty strings as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.map({True: True, False: False, "True": True, "False": False})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unrecognized boolean values: {bad}")
    return mapped.astype(bool)


def _short_model(name: str) -> str:
    return str(name).rsplit("/", 1)[-1]


def _base_run_mask(runs: pd.DataFrame) -> pd.Series:
    return (
        (runs["order_ablation"] == "a1")
        & (runs["adversary_ablation"] == "base")
        & (runs["heterogeneity_ablation"] == "h1")
    )


def build_enriched_turns(
    data: SweepData,
    *,
    base_only: bool = True,
) -> pd.DataFrame:
    """Build the canonical per-turn table used by every redux analysis.

    The returned frame contains pre-turn public/original histories, adversarial
    exposure, intended outcomes, and truthful-implementation payoffs.
    """
    turns = data.agent_metrics.copy()
    if turns.empty:
        return turns

    runs = data.runs.copy()
    if base_only:
        runs = runs[_base_run_mask(runs)].copy()

    run_ids = set(runs["run_id"])
    turns = turns[turns["run_id"].isin(run_ids)].copy()

    missing = [col for col in _RUN_METADATA_COLUMNS if col not in turns.columns]
    if missing:
        turns = turns.merge(
            runs[["run_id", *missing]].drop_duplicates("run_id"),
            on="run_id",
            how="left",
            validate="many_to_one",
        )

    for col in (
        "is_liar",
        "original_is_stag",
        "reported_is_stag",
        "was_flipped",
        "stag_success",
    ):
        if col in turns.columns:
            turns[col] = _coerce_bool(turns[col])

    turns["model_short"] = turns["model"].map(_short_model)
    turns["liar_share"] = turns["num_liars"] / turns["num_agents"]
    turns["honest"] = ~turns["is_liar"]
    turns["round"] = turns["round"].astype(int)
    turns["turn_index"] = turns["turn_index"].astype(int)
    turns = turns.sort_values(
        ["run_id", "round", "turn_index", "agent"],
        kind="stable",
    ).reset_index(drop=True)

    within_round = turns.groupby(["run_id", "round"], sort=False)
    turns["prior_public_stag_count"] = (
        within_round["reported_is_stag"]
        .transform(lambda s: s.astype(int).cumsum().shift(fill_value=0))
        .astype(int)
    )
    turns["prior_original_stag_count"] = (
        within_round["original_is_stag"]
        .transform(lambda s: s.astype(int).cumsum().shift(fill_value=0))
        .astype(int)
    )
    turns["prior_liar_count"] = (
        within_round["is_liar"]
        .transform(lambda s: s.astype(int).cumsum().shift(fill_value=0))
        .astype(int)
    )
    turns["n_observed"] = turns["turn_index"]
    observed = turns["n_observed"].replace(0, np.nan)
    turns["prior_public_stag_share"] = turns["prior_public_stag_count"] / observed
    turns["prior_original_stag_share"] = turns["prior_original_stag_count"] / observed
    turns["prior_liar_share"] = turns["prior_liar_count"] / observed
    turns["history_defined"] = turns["n_observed"] > 0

    round_history = (
        turns.groupby(["run_id", "round"], as_index=False)
        .agg(previous_public_stag_share=("reported_is_stag", "mean"))
        .sort_values(["run_id", "round"])
    )
    round_history["previous_round_public_stag_share"] = round_history.groupby(
        "run_id", sort=False
    )["previous_public_stag_share"].shift()
    turns = turns.merge(
        round_history[["run_id", "round", "previous_round_public_stag_share"]],
        on=["run_id", "round"],
        how="left",
        validate="many_to_one",
    )
    turns["previous_original_is_stag"] = turns.groupby(["run_id", "agent"], sort=False)[
        "original_is_stag"
    ].shift()

    q_star_by_game: dict[tuple[int, int, float, float, float], float] = {}
    q_star_values: list[float] = []
    for row in turns.itertuples(index=False):
        key = (
            int(row.num_agents),
            int(row.stag_success_threshold),
            float(row.payoff_stag_success),
            float(row.payoff_stag_fail),
            float(row.payoff_hare_fail),
        )
        if key not in q_star_by_game:
            q_star_by_game[key] = compute_q_star(
                num_agents=key[0],
                threshold_m=key[1],
                payoff_stag_success=key[2],
                payoff_stag_fail=key[3],
                payoff_hare_safe=key[4],
            )
        q_star_values.append(q_star_by_game[key])
    turns["q_star"] = q_star_values

    turns["public_history_prediction"] = (
        turns["prior_public_stag_share"] >= turns["q_star"]
    )
    turns["original_history_prediction"] = (
        turns["prior_original_stag_share"] >= turns["q_star"]
    )
    turns["public_history_match"] = turns["history_defined"] & (
        turns["public_history_prediction"] == turns["original_is_stag"]
    )
    turns["original_history_match"] = turns["history_defined"] & (
        turns["original_history_prediction"] == turns["original_is_stag"]
    )
    turns["public_history_brier"] = (
        turns["public_history_prediction"].astype(float)
        - turns["original_is_stag"].astype(float)
    ) ** 2
    turns["original_history_brier"] = (
        turns["original_history_prediction"].astype(float)
        - turns["original_is_stag"].astype(float)
    ) ** 2

    within_round = turns.groupby(["run_id", "round"], sort=False)
    turns["original_stag_count"] = within_round["original_is_stag"].transform("sum")
    turns["public_stag_count"] = within_round["reported_is_stag"].transform("sum")
    turns["intended_success"] = (
        turns["original_stag_count"] >= turns["stag_success_threshold"]
    )

    turns["truthful_payoff"] = np.select(
        [
            turns["intended_success"] & turns["original_is_stag"],
            turns["intended_success"] & ~turns["original_is_stag"],
            ~turns["intended_success"] & turns["original_is_stag"],
        ],
        [
            turns["payoff_stag_success"],
            turns["payoff_hare_when_stag_success"],
            turns["payoff_stag_fail"],
        ],
        default=turns["payoff_hare_fail"],
    ).astype(float)
    turns["mechanical_payoff_gap"] = turns["truthful_payoff"] - turns["realized_payoff"]
    turns["config_seed_cluster"] = (
        turns[
            [
                "model_short",
                "num_agents",
                "stag_success_threshold",
                "num_liars",
                "num_rounds",
                "seed",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    return turns


def build_action_channel_table(turns: pd.DataFrame) -> pd.DataFrame:
    """Collapse the enriched turn table to one row per run-round."""
    if turns.empty:
        return pd.DataFrame()

    frame = turns.copy()
    frame["honest_original_stag"] = frame["original_is_stag"].where(frame["honest"])
    frame["liar_original_stag"] = frame["original_is_stag"].where(frame["is_liar"])
    frame["honest_realized_payoff"] = frame["realized_payoff"].where(frame["honest"])
    frame["honest_truthful_payoff"] = frame["truthful_payoff"].where(frame["honest"])

    metadata = [
        "model",
        "model_short",
        "seed",
        "num_agents",
        "num_rounds",
        "num_liars",
        "liar_share",
        "stag_success_threshold",
        "order_ablation",
        "adversary_ablation",
        "heterogeneity_ablation",
    ]
    aggregations: dict[str, tuple[str, str]] = {col: (col, "first") for col in metadata}
    aggregations.update(
        {
            "original_stag_count": ("original_is_stag", "sum"),
            "public_stag_count": ("reported_is_stag", "sum"),
            "honest_original_stag_rate": ("honest_original_stag", "mean"),
            "liar_original_stag_rate": ("liar_original_stag", "mean"),
            "public_success": ("stag_success", "first"),
            "honest_realized_payoff": ("honest_realized_payoff", "mean"),
            "honest_truthful_payoff": ("honest_truthful_payoff", "mean"),
        }
    )
    channel = (
        frame.groupby(["run_id", "round"], as_index=False)
        .agg(**aggregations)
        .sort_values(["run_id", "round"])
        .reset_index(drop=True)
    )
    channel["intended_success"] = (
        channel["original_stag_count"] >= channel["stag_success_threshold"]
    )
    channel["shared_success"] = channel["intended_success"] & channel["public_success"]
    channel["flip_induced_loss"] = (
        channel["intended_success"] & ~channel["public_success"]
    )
    channel["flip_induced_rescue"] = (
        ~channel["intended_success"] & channel["public_success"]
    )
    channel["shared_failure"] = (
        ~channel["intended_success"] & ~channel["public_success"]
    )
    for col in (
        "public_success",
        "intended_success",
        "shared_success",
        "flip_induced_loss",
        "flip_induced_rescue",
        "shared_failure",
    ):
        channel[col] = channel[col].astype(float)
    channel["honest_mechanical_payoff_gap"] = (
        channel["honest_truthful_payoff"] - channel["honest_realized_payoff"]
    )
    return channel


def _cell_columns(frame: pd.DataFrame, group_cols: Sequence[str]) -> list[str]:
    cols = [col for col in _CELL_COLUMNS if col in frame.columns]
    for col in group_cols:
        if col not in cols:
            cols.append(col)
    return cols


def hierarchical_summary(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    bootstrap_replicates: int = _BOOTSTRAP_REPLICATES,
    seed: int = _BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Configuration-balanced, model-balanced summary with hierarchical CIs."""
    required = [*group_cols, *metrics, "model_short"]
    work = frame.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()

    cell_cols = _cell_columns(work, group_cols)
    cells = work.groupby(cell_cols, as_index=False)[list(metrics)].mean()
    model_group_cols = ["model_short", *group_cols]
    model_means = cells.groupby(model_group_cols, as_index=False)[list(metrics)].mean()
    summary = model_means.groupby(list(group_cols), as_index=False)[
        list(metrics)
    ].mean()
    model_counts = (
        model_means.groupby(list(group_cols), as_index=False)["model_short"]
        .nunique()
        .rename(columns={"model_short": "n_models"})
    )
    cell_counts = (
        cells.groupby(list(group_cols), as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    summary = summary.merge(model_counts, on=list(group_cols), how="left")
    summary = summary.merge(cell_counts, on=list(group_cols), how="left")

    rng = np.random.default_rng(seed)
    ci_rows: list[dict[str, object]] = []
    grouper: str | list[str] = (
        group_cols[0] if len(group_cols) == 1 else list(group_cols)
    )
    for group_key, group_cells in cells.groupby(grouper, sort=False, dropna=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        models = group_cells["model_short"].unique()
        boot = np.empty((bootstrap_replicates, len(metrics)), dtype=float)
        by_model = {
            model: group_cells[group_cells["model_short"] == model] for model in models
        }
        for b in range(bootstrap_replicates):
            sampled_models = rng.choice(models, size=len(models), replace=True)
            sampled_model_means: list[np.ndarray] = []
            for model in sampled_models:
                model_cells = by_model[model]
                sampled_indices = rng.integers(
                    0, len(model_cells), size=len(model_cells)
                )
                sampled = model_cells.iloc[sampled_indices]
                sampled_model_means.append(
                    sampled[list(metrics)].mean().to_numpy(dtype=float)
                )
            boot[b, :] = np.mean(sampled_model_means, axis=0)

        row: dict[str, object] = dict(zip(group_cols, key_tuple, strict=True))
        for idx, metric in enumerate(metrics):
            row[f"{metric}_low"] = float(np.quantile(boot[:, idx], 0.025))
            row[f"{metric}_high"] = float(np.quantile(boot[:, idx], 0.975))
        ci_rows.append(row)

    return summary.merge(pd.DataFrame(ci_rows), on=list(group_cols), how="left")


def configuration_summary(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    bootstrap_replicates: int = 600,
    seed: int = _BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Configuration-balanced summary, retaining model-specific groups.

    ``hierarchical_summary`` averages equally across models. This companion
    helper keeps any grouping columns, including ``model_short``, and averages
    repeated observations within configuration/seed cells before calculating
    the displayed group means and cell-bootstrap intervals.
    """
    required = [*group_cols, *metrics]
    work = frame.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()

    cell_cols = _cell_columns(work, group_cols)
    cells = work.groupby(cell_cols, as_index=False)[list(metrics)].mean()
    summary = cells.groupby(list(group_cols), as_index=False)[list(metrics)].mean()
    counts = (
        cells.groupby(list(group_cols), as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    summary = summary.merge(counts, on=list(group_cols), how="left")

    rng = np.random.default_rng(seed)
    ci_rows: list[dict[str, object]] = []
    grouper: str | list[str] = (
        group_cols[0] if len(group_cols) == 1 else list(group_cols)
    )
    for group_key, group_cells in cells.groupby(grouper, sort=False, dropna=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        values = group_cells[list(metrics)].to_numpy(dtype=float)
        sampled_indices = rng.integers(
            0,
            len(values),
            size=(bootstrap_replicates, len(values)),
        )
        boot = values[sampled_indices].mean(axis=1)
        row: dict[str, object] = dict(zip(group_cols, key_tuple, strict=True))
        for idx, metric in enumerate(metrics):
            row[f"{metric}_low"] = float(np.quantile(boot[:, idx], 0.025))
            row[f"{metric}_high"] = float(np.quantile(boot[:, idx], 0.975))
        ci_rows.append(row)

    return summary.merge(pd.DataFrame(ci_rows), on=list(group_cols), how="left")


def _error_arrays(summary: pd.DataFrame, metric: str) -> np.ndarray:
    values = summary[metric].to_numpy(dtype=float)
    lows = summary[f"{metric}_low"].to_numpy(dtype=float)
    highs = summary[f"{metric}_high"].to_numpy(dtype=float)
    return np.vstack([values - lows, highs - values])


def _percent_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def mechanism_decomposition_summary(channel: pd.DataFrame) -> pd.DataFrame:
    focal = channel[
        (channel["num_agents"] == _FOCAL_N)
        & (channel["stag_success_threshold"] == _FOCAL_M)
    ].copy()
    return hierarchical_summary(
        focal,
        group_cols=["liar_share"],
        metrics=[
            "honest_original_stag_rate",
            "intended_success",
            "public_success",
            "shared_success",
            "flip_induced_loss",
            "flip_induced_rescue",
            "shared_failure",
        ],
    ).sort_values("liar_share")


def all_mechanism_summary(channel: pd.DataFrame) -> pd.DataFrame:
    """Model-balanced decomposition for every N/M/F condition with support."""
    supported = channel.groupby(
        ["num_agents", "stag_success_threshold"], as_index=False
    )["liar_share"].nunique()
    supported_pairs = set(
        supported.loc[
            supported["liar_share"] >= 2,
            ["num_agents", "stag_success_threshold"],
        ].itertuples(index=False, name=None)
    )
    frame = channel[
        [
            (int(n), int(m)) in supported_pairs
            for n, m in zip(
                channel["num_agents"],
                channel["stag_success_threshold"],
                strict=False,
            )
        ]
    ].copy()
    return hierarchical_summary(
        frame,
        group_cols=["num_agents", "stag_success_threshold", "liar_share"],
        metrics=[
            "honest_original_stag_rate",
            "intended_success",
            "public_success",
            "flip_induced_loss",
            "shared_failure",
        ],
        bootstrap_replicates=600,
    ).sort_values(["num_agents", "stag_success_threshold", "liar_share"])


def fig_mechanism_facets(
    channel: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Appendix small multiples of intended and public success by N/M."""
    summary = all_mechanism_summary(channel)
    pairs = list(
        summary[["num_agents", "stag_success_threshold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    n_cols = 3
    n_rows = math.ceil(len(pairs) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.1 * n_rows),
        sharey=True,
        squeeze=False,
    )
    for ax, (n_agents, threshold) in zip(axes.flat, pairs, strict=False):
        subset = summary[
            (summary["num_agents"] == n_agents)
            & (summary["stag_success_threshold"] == threshold)
        ]
        x = subset["liar_share"].to_numpy(dtype=float)
        for metric, label, color in (
            ("intended_success", "Intended success", _INTENDED_COLOR),
            ("public_success", "Public success", _PUBLIC_COLOR),
        ):
            ax.errorbar(
                x,
                subset[metric],
                yerr=_error_arrays(subset, metric),
                color=color,
                label=label,
                marker="o",
                linewidth=1.8,
                capsize=2,
            )
        ax.axvline(
            1 - threshold / n_agents,
            color="0.4",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title(f"N={int(n_agents)}, M={int(threshold)}")
        ax.set_xticks(x, [f"{value:.0%}" for value in x])
        ax.set_xlabel("Corrupted fraction")
        _percent_axis(ax)
    for ax in axes.flat[len(pairs) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Success rate")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, summary


def fig_mechanism_decomposition(
    channel: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Focal analysis of honest choices and the two action layers."""
    summary = mechanism_decomposition_summary(channel)
    if summary.empty:
        raise ValueError("No N=5, M=3 base data found for mechanism decomposition")

    x = summary["liar_share"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    axes[0].errorbar(
        x,
        summary["honest_original_stag_rate"],
        yerr=_error_arrays(summary, "honest_original_stag_rate"),
        color=_HONEST_COLOR,
        marker="o",
        linewidth=2.2,
        capsize=3,
    )
    axes[0].set_ylabel("Original Stag rate")
    _percent_axis(axes[0])

    for metric, label, color in (
        ("intended_success", "Intended success", _INTENDED_COLOR),
        ("public_success", "Public success", _PUBLIC_COLOR),
    ):
        axes[1].errorbar(
            x,
            summary[metric],
            yerr=_error_arrays(summary, metric),
            label=label,
            color=color,
            marker="o",
            linewidth=2.2,
            capsize=3,
        )
    axes[1].axvline(
        1 - _FOCAL_M / _FOCAL_N,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label="Mechanical boundary",
    )
    axes[1].set_ylabel("Success rate")
    axes[1].legend(frameon=True, fontsize=_FONT_SIZE_LEGEND)
    _percent_axis(axes[1])

    for ax in axes:
        ax.set_xlabel("Corrupted-agent fraction")
        ax.set_xticks(x, [f"{value:.0%}" for value in x])
    fig.tight_layout()
    return fig, summary


def model_mechanism_summary(channel: pd.DataFrame) -> pd.DataFrame:
    """Configuration-balanced focal decomposition for each model."""
    focal = channel[
        (channel["num_agents"] == _FOCAL_N)
        & (channel["stag_success_threshold"] == _FOCAL_M)
    ].copy()
    return configuration_summary(
        focal,
        group_cols=["model_short", "liar_share"],
        metrics=[
            "honest_original_stag_rate",
            "intended_success",
            "public_success",
            "flip_induced_loss",
        ],
    ).sort_values(["model_short", "liar_share"])


def fig_model_mechanism_decomposition(
    channel: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Updated model breakdown of honest choices and the two outcome layers."""
    summary = model_mechanism_summary(channel)
    if summary.empty:
        raise ValueError("No N=5, M=3 base data found for model decomposition")

    models = sorted(summary["model_short"].unique())
    n_cols = 4
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(19, 4.4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    series = (
        (
            "honest_original_stag_rate",
            "Honest P(STAG)",
            _HONEST_COLOR,
            "o",
            "--",
        ),
        ("intended_success", "Intended success", _INTENDED_COLOR, "s", "-"),
        ("public_success", "Public success", _PUBLIC_COLOR, "^", "-"),
    )
    visible_axes: list[plt.Axes] = []
    for idx, (ax, model) in enumerate(zip(axes.flat, models, strict=False)):
        subset = summary[summary["model_short"] == model].sort_values("liar_share")
        x = subset["liar_share"].to_numpy(dtype=float)
        for metric, label, color, marker, linestyle in series:
            ax.errorbar(
                x,
                subset[metric],
                yerr=_error_arrays(subset, metric),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                capsize=2,
                label=label,
            )
        ax.axvline(
            1 - _FOCAL_M / _FOCAL_N,
            color="0.35",
            linestyle=":",
            linewidth=1.1,
            label="Mechanical boundary" if idx == 0 else None,
        )
        ax.set_title(model, fontsize=_FONT_SIZE_FACET)
        ax.set_xticks(x, [f"{value:.0%}" for value in x])
        _percent_axis(ax)
        visible_axes.append(ax)

    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for row in range(n_rows):
        axes[row, 0].set_ylabel("Rate")
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Corrupted-agent fraction")

    handles, labels = visible_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, summary


def coordination_dynamics_summary(channel: pd.DataFrame) -> pd.DataFrame:
    """Round dynamics for complete four-round focal games only."""
    focal = channel[
        (channel["num_agents"] == _FOCAL_N)
        & (channel["stag_success_threshold"] == _FOCAL_M)
        & (channel["num_rounds"] == 4)
    ].copy()
    return hierarchical_summary(
        focal,
        group_cols=["liar_share", "round"],
        metrics=[
            "honest_original_stag_rate",
            "intended_success",
            "public_success",
        ],
        bootstrap_replicates=600,
    ).sort_values(["liar_share", "round"])


def fig_coordination_dynamics_redux(
    channel: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Updated round plot separating honest choices and outcome layers."""
    summary = coordination_dynamics_summary(channel)
    if summary.empty:
        raise ValueError("No complete four-round N=5, M=3 base data found")

    shares = sorted(summary["liar_share"].unique())
    fig, axes = plt.subplots(
        2,
        len(shares),
        figsize=(4.6 * len(shares), 7.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for col, share in enumerate(shares):
        subset = summary[np.isclose(summary["liar_share"], share)].sort_values("round")
        rounds = subset["round"].to_numpy(dtype=int)
        axes[0, col].errorbar(
            rounds,
            subset["honest_original_stag_rate"],
            yerr=_error_arrays(subset, "honest_original_stag_rate"),
            color=_HONEST_COLOR,
            marker="o",
            linestyle="-",
            linewidth=2,
            capsize=2,
        )
        axes[0, col].set_title(f"{share:.0%} corrupted")
        _percent_axis(axes[0, col])

        for metric, label, series_color, series_marker, series_style in (
            ("intended_success", "Intended success", _INTENDED_COLOR, "o", "-"),
            ("public_success", "Public success", _PUBLIC_COLOR, "s", "--"),
        ):
            axes[1, col].errorbar(
                rounds,
                subset[metric],
                yerr=_error_arrays(subset, metric),
                color=series_color,
                marker=series_marker,
                linestyle=series_style,
                linewidth=2,
                capsize=2,
                label=label,
            )
        axes[1, col].set_xticks(rounds)
        axes[1, col].set_xlabel("Round")
        _percent_axis(axes[1, col])

    axes[0, 0].set_ylabel("Honest P(STAG)")
    axes[1, 0].set_ylabel("Success rate")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, summary


def speaking_position_summary(turns: pd.DataFrame) -> pd.DataFrame:
    """Honest action rates by fixed speaking position for each focal model."""
    eligible = turns[
        turns["honest"]
        & (turns["num_agents"] == _FOCAL_N)
        & (turns["stag_success_threshold"] == _FOCAL_M)
    ].copy()
    return configuration_summary(
        eligible,
        group_cols=["model_short", "liar_share", "turn_index"],
        metrics=["original_is_stag"],
    ).sort_values(["model_short", "liar_share", "turn_index"])


def fig_speaking_position_redux(
    turns: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Updated descriptive turn-order plot using honest Stag choices."""
    summary = speaking_position_summary(turns)
    if summary.empty:
        raise ValueError("No focal honest turns found for speaking-position plot")

    models = sorted(summary["model_short"].unique())
    shares = sorted(summary["liar_share"].unique())
    n_cols = 4
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(19, 4.4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    visible_axes: list[plt.Axes] = []
    for ax, model in zip(axes.flat, models, strict=False):
        model_rows = summary[summary["model_short"] == model]
        for share in shares:
            subset = model_rows[
                np.isclose(model_rows["liar_share"], share)
            ].sort_values("turn_index")
            if subset.empty:
                continue
            color, marker, linestyle = _CORRUPTION_STYLES[round(float(share), 1)]
            positions = subset["turn_index"].to_numpy(dtype=int) + 1
            ax.errorbar(
                positions,
                subset["original_is_stag"],
                yerr=_error_arrays(subset, "original_is_stag"),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                capsize=2,
                label=f"{share:.0%}",
            )
        ax.set_title(model, fontsize=_FONT_SIZE_FACET)
        ax.set_xticks(range(1, _FOCAL_N + 1))
        _percent_axis(ax)
        visible_axes.append(ax)

    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for row in range(n_rows):
        axes[row, 0].set_ylabel("Honest P(STAG)")
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Speaking position")

    handles, labels = visible_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Corrupted-agent fraction",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(shares),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, summary


def public_history_response_summary(
    data: SweepData,
    *,
    base_run_ids: set[str],
) -> pd.DataFrame:
    """Model-specific honest responses under three public-history rules."""
    frames = [
        build_belief_benchmark(data),
        build_carryover_benchmark(data),
        build_trust_weighted_benchmark(data),
    ]
    eligible = pd.concat(frames, ignore_index=True)
    eligible = eligible[
        eligible["run_id"].isin(base_run_ids)
        & (eligible["role"] == "Honest")
        & eligible["benchmark_defined"]
    ].copy()
    eligible["public_history_margin"] = eligible["benchmark_q"] - eligible["q_star"]
    margin_bins = pd.cut(
        eligible["public_history_margin"],
        bins=_PUBLIC_MARGIN_BIN_EDGES,
        include_lowest=True,
    )
    eligible["public_history_margin_mid"] = margin_bins.map(
        lambda interval: interval.mid if pd.notna(interval) else np.nan
    ).astype(float)
    summary = configuration_summary(
        eligible,
        group_cols=[
            "model_short",
            "benchmark_rule",
            "public_history_margin_mid",
        ],
        metrics=["original_is_stag"],
    )
    return summary[summary["n_cells"] >= 3].sort_values(
        ["model_short", "benchmark_rule", "public_history_margin_mid"]
    )


def fig_public_history_response_redux(
    data: SweepData,
    *,
    base_run_ids: set[str],
) -> tuple[plt.Figure, pd.DataFrame]:
    """Honest action curves under three public-history benchmark rules."""
    summary = public_history_response_summary(data, base_run_ids=base_run_ids)
    if summary.empty:
        raise ValueError("No eligible public-history response bins found")

    models = sorted(summary["model_short"].unique())
    n_cols = 4
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(19, 4.4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    visible_axes: list[plt.Axes] = []
    for idx, (ax, model) in enumerate(zip(axes.flat, models, strict=False)):
        model_rows = summary[summary["model_short"] == model]
        for rule, (color, marker, linestyle) in _BENCHMARK_RULE_STYLES.items():
            subset = model_rows[model_rows["benchmark_rule"] == rule].sort_values(
                "public_history_margin_mid"
            )
            if subset.empty:
                continue
            ax.errorbar(
                subset["public_history_margin_mid"],
                subset["original_is_stag"],
                yerr=_error_arrays(subset, "original_is_stag"),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                capsize=2,
                label=rule if idx == 0 else None,
            )
        ax.axvline(
            0,
            color="0.35",
            linestyle=":",
            linewidth=1.1,
            label=r"Rule margin $=0$" if idx == 0 else None,
        )
        ax.set_title(model, fontsize=_FONT_SIZE_FACET)
        ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
        _percent_axis(ax)
        visible_axes.append(ax)

    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for row in range(n_rows):
        axes[row, 0].set_ylabel("Honest P(STAG)")
    handles, labels = visible_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
    )
    fig.supxlabel(r"Public-history margin $\hat{q}_{rule} - q^*$", y=0.025)
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.13,
        top=0.82,
        wspace=0.12,
        hspace=0.42,
    )
    return fig, summary


def history_benchmark_summary(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = turns[
        turns["honest"]
        & turns["history_defined"]
        & (turns["num_agents"] == _FOCAL_N)
        & (turns["stag_success_threshold"] == _FOCAL_M)
    ].copy()
    summary = hierarchical_summary(
        eligible,
        group_cols=["liar_share"],
        metrics=[
            "public_history_match",
            "original_history_match",
            "public_history_brier",
            "original_history_brier",
        ],
    ).sort_values("liar_share")

    cell_cols = _cell_columns(eligible, ["liar_share"])
    per_cell = eligible.groupby(cell_cols, as_index=False).agg(
        n_turns=("original_is_stag", "size"),
        public_history_match=("public_history_match", "mean"),
        original_history_match=("original_history_match", "mean"),
        public_history_brier=("public_history_brier", "mean"),
        original_history_brier=("original_history_brier", "mean"),
    )
    per_cell["match_advantage_public"] = (
        per_cell["public_history_match"] - per_cell["original_history_match"]
    )
    return summary, per_cell


def fig_public_vs_original_history(
    turns: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Compare the transcript agents saw with the hidden original-action history."""
    summary, per_cell = history_benchmark_summary(turns)
    if summary.empty:
        raise ValueError("No eligible N=5, M=3 honest turns found")

    x = summary["liar_share"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for metric, label, color in (
        ("public_history_match", "Public history", _PUBLIC_COLOR),
        ("original_history_match", "Hidden original history", _ORIGINAL_HISTORY_COLOR),
    ):
        axes[0].errorbar(
            x,
            summary[metric],
            yerr=_error_arrays(summary, metric),
            label=label,
            color=color,
            marker="o",
            linewidth=2.2,
            capsize=3,
        )
    axes[0].set_xlabel("Corrupted-agent fraction")
    axes[0].set_ylabel("Honest-action match rate", labelpad=8)
    axes[0].set_xticks(x, [f"{value:.0%}" for value in x])
    axes[0].legend(frameon=True, fontsize=_FONT_SIZE_LEGEND)
    _percent_axis(axes[0])

    model_f = per_cell.groupby(["model_short", "liar_share"], as_index=False)[
        "match_advantage_public"
    ].mean()
    high = model_f[model_f["liar_share"] >= 0.6].copy()
    order = sorted(high["model_short"].unique())
    y_positions = {model: idx for idx, model in enumerate(order)}
    marker_by_share = {
        0.6: ("o", "#DD8452"),
        0.8: ("s", "#C44E52"),
    }
    for share, (marker, color) in marker_by_share.items():
        share_rows = high[np.isclose(high["liar_share"], share)]
        axes[1].scatter(
            share_rows["match_advantage_public"],
            [y_positions[model] for model in share_rows["model_short"]],
            marker=marker,
            color=color,
            s=70,
            label=f"{share:.0%} corrupted",
            zorder=3,
        )
    axes[1].axvline(0, color="0.35", linewidth=1.2)
    axes[1].set_yticks(range(len(order)), order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Match-rate difference")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        fontsize=_FONT_SIZE_LEGEND,
    )
    fig.tight_layout(rect=(0.025, 0, 1, 0.92))
    return fig, summary, per_cell


def fit_clustered_lpm(
    frame: pd.DataFrame,
    *,
    predictors: Sequence[str],
    specification: str,
) -> pd.DataFrame:
    """Fit an LPM with structural fixed effects and cluster-robust covariance."""
    required = ["original_is_stag", "config_seed_cluster", *predictors]
    work = frame.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()

    work["game_condition"] = (
        work[
            [
                "num_agents",
                "stag_success_threshold",
                "num_liars",
                "num_rounds",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    controls = pd.get_dummies(
        work[["model_short", "game_condition", "round", "turn_index"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    predictor_frame = work[list(predictors)].astype(float).reset_index(drop=True)
    design = pd.concat([predictor_frame, controls.reset_index(drop=True)], axis=1)
    design.insert(0, "intercept", 1.0)
    x = design.to_numpy(dtype=float)
    y = work["original_is_stag"].astype(float).to_numpy()

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta

    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    clusters = work["config_seed_cluster"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = xtx_inv @ meat @ xtx_inv
    n, k = x.shape
    g = len(unique_clusters)
    if g > 1 and n > k:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - k))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))

    rows: list[dict[str, float | int | str]] = []
    for predictor in predictors:
        idx = design.columns.get_loc(predictor)
        estimate = float(beta[idx])
        se = float(standard_errors[idx])
        z = estimate / se if se > 0 else float("nan")
        p = float(2 * norm.sf(abs(z))) if math.isfinite(z) else float("nan")
        rows.append(
            {
                "specification": specification,
                "predictor": predictor,
                "estimate": estimate,
                "cluster_se": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "p_value": p,
                "n_turns": n,
                "n_clusters": g,
            }
        )
    return pd.DataFrame(rows)


def sequential_exposure_analysis(
    turns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = turns[turns["honest"] & turns["history_defined"]].copy()
    specifications = [
        fit_clustered_lpm(
            eligible,
            predictors=["prior_liar_share"],
            specification="Exposure only",
        ),
        fit_clustered_lpm(
            eligible,
            predictors=["prior_public_stag_share"],
            specification="Public history only",
        ),
        fit_clustered_lpm(
            eligible,
            predictors=["prior_liar_share", "prior_public_stag_share"],
            specification="Exposure + public history",
        ),
    ]
    coefficients = pd.concat(specifications, ignore_index=True)

    eligible["round_stage"] = np.where(
        eligible["round"] == 1, "Round 1", "Later rounds"
    )
    bin_edges = np.array([-0.001, 0.25, 0.50, 0.75, 1.001])
    eligible["public_history_bin"] = pd.cut(
        eligible["prior_public_stag_share"],
        bins=bin_edges,
        labels=[0.125, 0.375, 0.625, 0.875],
        include_lowest=True,
    ).astype(float)
    response = hierarchical_summary(
        eligible,
        group_cols=["round_stage", "public_history_bin"],
        metrics=["original_is_stag"],
        bootstrap_replicates=600,
    ).sort_values(["round_stage", "public_history_bin"])

    negative = turns[
        turns["honest"] & (turns["round"] == 1) & (turns["turn_index"] == 0)
    ].copy()
    negative_control = hierarchical_summary(
        negative,
        group_cols=["num_agents", "stag_success_threshold", "liar_share"],
        metrics=["original_is_stag"],
        bootstrap_replicates=600,
    )
    return coefficients, response, negative_control


def fig_sequential_exposure(
    turns: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients, response, negative = sequential_exposure_analysis(turns)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    stage_colors = {"Round 1": _PUBLIC_COLOR, "Later rounds": _HONEST_COLOR}
    for stage in ("Round 1", "Later rounds"):
        subset = response[response["round_stage"] == stage]
        if subset.empty:
            continue
        axes[0].errorbar(
            subset["public_history_bin"],
            subset["original_is_stag"],
            yerr=_error_arrays(subset, "original_is_stag"),
            color=stage_colors[stage],
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=stage,
        )
    axes[0].set_xlabel("Prior public Stag share (binned)")
    axes[0].set_ylabel("Honest P(STAG)")
    axes[0].set_xticks(
        [0.125, 0.375, 0.625, 0.875], ["0-25%", "25-50%", "50-75%", "75-100%"]
    )
    axes[0].legend(frameon=True, fontsize=_FONT_SIZE_LEGEND)
    _percent_axis(axes[0])

    coefficient_plot = coefficients.copy()
    specification_names = {
        "Exposure only": "Exposure only",
        "Public history only": "Public only",
        "Exposure + public history": "Combined",
    }
    predictor_names = {
        "prior_liar_share": "corrupted share",
        "prior_public_stag_share": "public Stag share",
    }
    coefficient_plot["label"] = coefficient_plot.apply(
        lambda row: (
            f"{specification_names[row['specification']]}: "
            f"{predictor_names[row['predictor']]}"
        ),
        axis=1,
    )
    coefficient_plot = coefficient_plot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(coefficient_plot))
    axes[1].errorbar(
        coefficient_plot["estimate"],
        y,
        xerr=np.vstack(
            [
                coefficient_plot["estimate"] - coefficient_plot["ci_low"],
                coefficient_plot["ci_high"] - coefficient_plot["estimate"],
            ]
        ),
        fmt="o",
        color=_INTENDED_COLOR,
        capsize=3,
    )
    axes[1].axvline(0, color="0.35", linewidth=1.2)
    axes[1].set_yticks(y, coefficient_plot["label"])
    axes[1].set_xlabel("Effect on honest P(STAG)")
    axes[1].xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    fig.tight_layout()
    return fig, coefficients, response, negative


def welfare_summary(channel: pd.DataFrame) -> pd.DataFrame:
    focal = channel[
        (channel["num_agents"] == _FOCAL_N)
        & (channel["stag_success_threshold"] == _FOCAL_M)
    ].copy()
    return hierarchical_summary(
        focal,
        group_cols=["liar_share"],
        metrics=[
            "honest_realized_payoff",
            "honest_truthful_payoff",
            "honest_mechanical_payoff_gap",
        ],
    ).sort_values("liar_share")


def fig_welfare_decomposition(
    channel: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    summary = welfare_summary(channel)
    if summary.empty:
        raise ValueError("No N=5, M=3 base data found for welfare decomposition")

    x = summary["liar_share"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for metric, label, color in (
        (
            "honest_truthful_payoff",
            "Payoff under selected actions",
            _INTENDED_COLOR,
        ),
        ("honest_realized_payoff", "Realized public-action payoff", _PUBLIC_COLOR),
    ):
        ax.errorbar(
            x,
            summary[metric],
            yerr=_error_arrays(summary, metric),
            color=color,
            label=label,
            marker="o",
            linewidth=2.2,
            capsize=3,
        )
    ax.fill_between(
        x,
        summary["honest_realized_payoff"].to_numpy(dtype=float),
        summary["honest_truthful_payoff"].to_numpy(dtype=float),
        color=_PUBLIC_COLOR,
        alpha=0.12,
        label="Payoff difference",
    )
    ax.set_xticks(x, [f"{value:.0%}" for value in x])
    ax.set_xlabel("Corrupted-agent fraction")
    ax.set_ylabel("Mean honest-agent payoff")
    ax.legend(frameon=True, fontsize=_FONT_SIZE_LEGEND)
    ax.set_ylim(0, max(4.15, float(summary["honest_truthful_payoff_high"].max()) + 0.1))
    fig.tight_layout()
    return fig, summary


def build_b3_matched_decomposition(channel_all: pd.DataFrame) -> pd.DataFrame:
    """Configuration-match base and b3 after collapsing repeated base runs."""
    subset = channel_all[
        (channel_all["order_ablation"] == "a1")
        & (channel_all["heterogeneity_ablation"] == "h1")
        & channel_all["adversary_ablation"].isin(["base", "b3"])
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    pair_cols = [
        "model_short",
        "seed",
        "num_agents",
        "num_rounds",
        "num_liars",
        "stag_success_threshold",
        "round",
    ]
    metrics = [
        "honest_original_stag_rate",
        "intended_success",
        "public_success",
        "flip_induced_loss",
        "flip_induced_rescue",
        "honest_realized_payoff",
        "honest_truthful_payoff",
    ]
    collapsed = subset.groupby([*pair_cols, "adversary_ablation"], as_index=False)[
        metrics
    ].mean()
    base = collapsed[collapsed["adversary_ablation"] == "base"].drop(
        columns="adversary_ablation"
    )
    b3 = collapsed[collapsed["adversary_ablation"] == "b3"].drop(
        columns="adversary_ablation"
    )
    pairs = base.merge(
        b3,
        on=pair_cols,
        how="inner",
        suffixes=("_base", "_b3"),
        validate="one_to_one",
    )
    if pairs.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for round_num, round_pairs in pairs.groupby("round", sort=True):
        for metric in metrics:
            differences = round_pairs[f"{metric}_b3"] - round_pairs[f"{metric}_base"]
            se = float(differences.sem()) if len(differences) > 1 else float("nan")
            rows.append(
                {
                    "round": int(round_num),
                    "metric": metric,
                    "matched_pairs": int(len(round_pairs)),
                    "base_mean": float(round_pairs[f"{metric}_base"].mean()),
                    "b3_mean": float(round_pairs[f"{metric}_b3"].mean()),
                    "difference_b3_minus_base": float(differences.mean()),
                    "paired_se": se,
                    "ci_low": float(differences.mean() - 1.96 * se),
                    "ci_high": float(differences.mean() + 1.96 * se),
                }
            )
    return pd.DataFrame(rows)


def build_coverage_table(data: SweepData) -> pd.DataFrame:
    runs = data.runs[_base_run_mask(data.runs)].copy()
    runs["model_short"] = runs["model"].map(_short_model)
    cell_cols = [
        "model_short",
        "num_agents",
        "stag_success_threshold",
        "num_liars",
        "num_rounds",
        "seed",
    ]
    unique_cells = runs[cell_cols].drop_duplicates()
    coverage = runs.groupby(
        ["model_short", "num_agents", "stag_success_threshold"],
        as_index=False,
    ).agg(
        n_runs=("run_id", "size"),
        liar_counts=("num_liars", lambda s: ",".join(map(str, sorted(s.unique())))),
        round_counts=("num_rounds", lambda s: ",".join(map(str, sorted(s.unique())))),
        n_seeds=("seed", "nunique"),
    )
    cell_counts = (
        unique_cells.groupby(
            ["model_short", "num_agents", "stag_success_threshold"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_unique_config_seed_cells"})
    )
    return coverage.merge(
        cell_counts,
        on=["model_short", "num_agents", "stag_success_threshold"],
        how="left",
    )


def _format_markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    """Small dependency-free Markdown table formatter."""
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for col in display.columns:
        display[col] = display[col].map(
            lambda value: (
                ""
                if pd.isna(value)
                else str(int(value))
                if isinstance(value, numbers.Integral) and not isinstance(value, bool)
                else f"{float(value):.{decimals}f}"
                if isinstance(value, numbers.Real) and not isinstance(value, bool)
                else value
            )
        )
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _summary_markdown(
    mechanism: pd.DataFrame,
    history: pd.DataFrame,
    welfare: pd.DataFrame,
    coefficients: pd.DataFrame,
    b3: pd.DataFrame,
) -> str:
    mechanism_display = mechanism[
        [
            "liar_share",
            "honest_original_stag_rate",
            "intended_success",
            "public_success",
            "flip_induced_loss",
        ]
    ].copy()
    mechanism_display["liar_share"] = mechanism_display["liar_share"].map(
        lambda value: f"{value:.0%}"
    )
    history_display = history[
        ["liar_share", "public_history_match", "original_history_match"]
    ].copy()
    history_display["liar_share"] = history_display["liar_share"].map(
        lambda value: f"{value:.0%}"
    )
    welfare_display = welfare[
        [
            "liar_share",
            "honest_realized_payoff",
            "honest_truthful_payoff",
            "honest_mechanical_payoff_gap",
        ]
    ].copy()
    welfare_display["liar_share"] = welfare_display["liar_share"].map(
        lambda value: f"{value:.0%}"
    )

    return f"""# Redux Analysis Summary

This directory contains the mechanism-focused reanalysis of
`logs/all_combination`. It was generated entirely from existing CSV logs; no
new model calls or simulations were made.

## Focal mechanism decomposition: N=5, M=3

{_format_markdown_table(mechanism_display)}

`intended_success` applies the threshold to recorded pre-flip choices.
`public_success` is the executed outcome in the original simulation.
`flip_induced_loss` is the share of run-rounds where intended choices succeed
but public actions fail.

## Public versus hidden-original history

{_format_markdown_table(history_display)}

The public-history benchmark uses only reports available to the deciding
agent. The hidden-original benchmark is an analyst-only contrast.

## Honest welfare decomposition

{_format_markdown_table(welfare_display)}

The truthful payoff applies the payoff rule to the same logged pre-flip
choices. It is not a no-corruption transcript counterfactual.

## Sequential response models

{_format_markdown_table(coefficients)}

These are linear probability models with model, game-condition, round, and
speaking-position fixed effects. Standard errors are clustered by
model/configuration/seed.

## Public-history response rules

Figure 14 compares naive within-round aggregation, cross-round carryover, and
trust-weighted public-history estimates. The rules change $\\hat{{q}}$; the game
threshold $q^*$ remains fixed for a given payoff/threshold configuration.

## Random-noise robustness

The matched base-versus-b3 table contains {len(b3):,} round/metric rows. See
`tables/b3_matched_decomposition.csv` for the complete results.

## Files

- `fig1_mechanism_decomposition`: honest choices and intended versus public success.
- `fig1b_mechanism_by_model`: the same three measures separated by model.
- `fig2_public_vs_original_history`: which action history better matches honest decisions.
- `fig3_sequential_exposure`: response curves and fixed-effect exposure estimates.
- `fig4_welfare_decomposition`: realized versus truthful-implementation honest payoff.
- `fig9_coordination_dynamics_redux`: honest choices and both outcome layers by round.
- `fig10_speaking_position_redux`: descriptive honest Stag choices by fixed position.
- `fig14_public_history_response_redux`: model-specific response under naive, carryover, and trust-weighted public-history rules.
- `figA1_mechanism_facets`: intended/public outcomes across all supported N/M games.
- `tables/`: machine-readable summaries, model estimates, coverage, and robustness checks.
- `reproducibility_appendix.md`: prompts and implementation details to include in the paper appendix.
- `pdf/`: vector versions of all figures.

In Figure 10, a missing position/share point means that no honest agent occupied
that fixed speaking position in the supported runs; it is not a zero-valued
observation or a rendering failure.
"""


def _reproducibility_appendix(logs_dir: Path, coverage: pd.DataFrame) -> str:
    lie_prompt = (Path(__file__).parent / "prompts" / "justify_lie.jinja").read_text(
        encoding="utf-8"
    )
    n_runs = int(pd.read_csv(logs_dir / "stag_hunt_runs.csv").shape[0])
    return f"""# Reproducibility Appendix Draft

## Agent system prompt

```text
{AGENT_SYSTEM_PROMPT}
```

## Structured output schema

```text
reported_action: one of STAG or HARE
confidence: number in [0, 1]
justification: one sentence
```

## Round-start message

```text
Round {{round_num}} begins. Based on the public reports from others, what
action do you choose? Remember: N={{num_agents}}, M={{threshold_m}}.
```

## Corrupted-justification prompt

```text
{lie_prompt.rstrip()}
```

## Transcript and corruption procedure

- Public entries use `Agent_name: {{structured JSON decision}}`.
- In the base condition, a corrupted reporter's action is deterministically inverted.
- In `b3`, the public action is sampled uniformly from STAG/HARE.
- When the action changes, the justification is regenerated with temperature 1.0.
- The original confidence is retained after the action and justification change.
- Outcomes and realized payoffs are calculated from public, post-change actions.
- The original and public actions are both retained in the agent-metrics CSV.

## Data coverage

The combined runs index contains {n_runs:,} runs before the base-condition
filter. The exact base coverage is in `tables/data_coverage.csv` ({len(coverage):,}
model/N/M rows). The analysis does not describe this as a complete factorial
grid because the realized support is unbalanced.
"""


def _save_figure_formats(
    fig: plt.Figure,
    *,
    output_dir: Path,
    name: str,
    formats: Iterable[str],
) -> list[Path]:
    saved: list[Path] = []
    for fmt in formats:
        target_dir = output_dir if fmt == "png" else output_dir / fmt
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{name}.{fmt}"
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=_SAVE_PAD_INCHES,
        )
        saved.append(path)
    plt.close(fig)
    return saved


def generate_redux_analysis(
    logs_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Generate the complete redux suite in a separate output directory."""
    _apply_redux_style()
    logs_path = Path(logs_dir)
    out = (
        Path(output_dir)
        if output_dir is not None
        else Path("output") / f"{logs_path.name}_redux"
    )
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_sweep_data(logs_path)
    base_turns = build_enriched_turns(data, base_only=True)
    all_turns = build_enriched_turns(data, base_only=False)
    base_channel = build_action_channel_table(base_turns)
    all_channel = build_action_channel_table(all_turns)

    saved: list[Path] = []

    fig1, mechanism = fig_mechanism_decomposition(base_channel)
    saved.extend(
        _save_figure_formats(
            fig1,
            output_dir=out,
            name="fig1_mechanism_decomposition",
            formats=formats,
        )
    )
    mechanism_path = tables_dir / "mechanism_decomposition.csv"
    mechanism.to_csv(mechanism_path, index=False)
    saved.append(mechanism_path)

    fig1b, mechanism_by_model = fig_model_mechanism_decomposition(base_channel)
    saved.extend(
        _save_figure_formats(
            fig1b,
            output_dir=out,
            name="fig1b_mechanism_by_model",
            formats=formats,
        )
    )
    mechanism_by_model_path = tables_dir / "mechanism_by_model.csv"
    mechanism_by_model.to_csv(mechanism_by_model_path, index=False)
    saved.append(mechanism_by_model_path)

    fig_a1, mechanism_all = fig_mechanism_facets(base_channel)
    saved.extend(
        _save_figure_formats(
            fig_a1,
            output_dir=out,
            name="figA1_mechanism_facets",
            formats=formats,
        )
    )
    mechanism_all_path = tables_dir / "mechanism_all_supported_games.csv"
    mechanism_all.to_csv(mechanism_all_path, index=False)
    saved.append(mechanism_all_path)

    fig2, history, history_cells = fig_public_vs_original_history(base_turns)
    saved.extend(
        _save_figure_formats(
            fig2,
            output_dir=out,
            name="fig2_public_vs_original_history",
            formats=formats,
        )
    )
    history_path = tables_dir / "history_benchmark_summary.csv"
    history.to_csv(history_path, index=False)
    saved.append(history_path)
    history_cells_path = tables_dir / "history_benchmark_cells.csv"
    history_cells.to_csv(history_cells_path, index=False)
    saved.append(history_cells_path)
    history_eligible = base_turns[
        base_turns["honest"] & base_turns["history_defined"]
    ].copy()
    history_all = hierarchical_summary(
        history_eligible,
        group_cols=["num_agents", "stag_success_threshold", "liar_share"],
        metrics=[
            "public_history_match",
            "original_history_match",
            "public_history_brier",
            "original_history_brier",
        ],
        bootstrap_replicates=600,
    )
    history_all_path = tables_dir / "history_benchmark_all_games.csv"
    history_all.to_csv(history_all_path, index=False)
    saved.append(history_all_path)

    fig3, coefficients, response, negative = fig_sequential_exposure(base_turns)
    saved.extend(
        _save_figure_formats(
            fig3,
            output_dir=out,
            name="fig3_sequential_exposure",
            formats=formats,
        )
    )
    for name, frame in (
        ("sequential_lpm", coefficients),
        ("sequential_response_bins", response),
        ("round1_first_speaker_negative_control", negative),
    ):
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved.append(path)

    fig9, dynamics = fig_coordination_dynamics_redux(base_channel)
    saved.extend(
        _save_figure_formats(
            fig9,
            output_dir=out,
            name="fig9_coordination_dynamics_redux",
            formats=formats,
        )
    )
    dynamics_path = tables_dir / "coordination_dynamics_by_round.csv"
    dynamics.to_csv(dynamics_path, index=False)
    saved.append(dynamics_path)

    fig10, speaking_position = fig_speaking_position_redux(base_turns)
    saved.extend(
        _save_figure_formats(
            fig10,
            output_dir=out,
            name="fig10_speaking_position_redux",
            formats=formats,
        )
    )
    speaking_position_path = tables_dir / "speaking_position_by_model.csv"
    speaking_position.to_csv(speaking_position_path, index=False)
    saved.append(speaking_position_path)

    fig14, public_history_response = fig_public_history_response_redux(
        data,
        base_run_ids=set(base_turns["run_id"]),
    )
    saved.extend(
        _save_figure_formats(
            fig14,
            output_dir=out,
            name="fig14_public_history_response_redux",
            formats=formats,
        )
    )
    public_history_response_path = tables_dir / "public_history_response_by_model.csv"
    public_history_response.to_csv(public_history_response_path, index=False)
    saved.append(public_history_response_path)

    fig4, welfare = fig_welfare_decomposition(base_channel)
    saved.extend(
        _save_figure_formats(
            fig4,
            output_dir=out,
            name="fig4_welfare_decomposition",
            formats=formats,
        )
    )
    welfare_path = tables_dir / "welfare_decomposition.csv"
    welfare.to_csv(welfare_path, index=False)
    saved.append(welfare_path)
    welfare_all = hierarchical_summary(
        base_channel,
        group_cols=["num_agents", "stag_success_threshold", "liar_share"],
        metrics=[
            "honest_realized_payoff",
            "honest_truthful_payoff",
            "honest_mechanical_payoff_gap",
        ],
        bootstrap_replicates=600,
    )
    welfare_all_path = tables_dir / "welfare_all_games.csv"
    welfare_all.to_csv(welfare_all_path, index=False)
    saved.append(welfare_all_path)

    b3 = build_b3_matched_decomposition(all_channel)
    b3_path = tables_dir / "b3_matched_decomposition.csv"
    b3.to_csv(b3_path, index=False)
    saved.append(b3_path)

    coverage = build_coverage_table(data)
    coverage_path = tables_dir / "data_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    saved.append(coverage_path)

    channel_path = tables_dir / "action_channel_run_round.csv"
    base_channel.to_csv(channel_path, index=False)
    saved.append(channel_path)
    history_columns = [
        "run_id",
        "round",
        "turn_index",
        "agent",
        "model_short",
        "seed",
        "num_agents",
        "stag_success_threshold",
        "num_liars",
        "num_rounds",
        "is_liar",
        "original_is_stag",
        "reported_is_stag",
        "n_observed",
        "prior_liar_count",
        "prior_liar_share",
        "prior_public_stag_share",
        "prior_original_stag_share",
        "q_star",
        "public_history_match",
        "original_history_match",
        "intended_success",
        "truthful_payoff",
        "realized_payoff",
    ]
    histories_path = tables_dir / "enriched_turn_histories.csv"
    base_turns[history_columns].to_csv(histories_path, index=False)
    saved.append(histories_path)

    summary_path = out / "analysis_summary.md"
    summary_path.write_text(
        _summary_markdown(mechanism, history, welfare, coefficients, b3),
        encoding="utf-8",
    )
    saved.append(summary_path)

    appendix_path = out / "reproducibility_appendix.md"
    appendix_path.write_text(
        _reproducibility_appendix(logs_path, coverage),
        encoding="utf-8",
    )
    saved.append(appendix_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the mechanism-focused Stag Hunt redux analysis.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/all_combination",
        help="Directory containing the combined sweep CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to output/<logs-name>_redux.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help="Figure formats to generate (default: png pdf).",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output") / f"{Path(args.logs_dir).name}_redux"
    )
    saved = generate_redux_analysis(
        logs_dir=args.logs_dir,
        output_dir=output_dir,
        formats=args.formats,
    )
    print(f"Generated {len(saved)} redux artifacts in {output_dir}")


if __name__ == "__main__":
    main()
