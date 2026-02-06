"""Reusable results/provenance helpers for simulation runs."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any, Sequence


def get_git_commit() -> str:
    """Best-effort lookup of the current git commit hash."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def build_run_id(logger: Any | None, prefix: str = "stag_hunt_simulation") -> str:
    """Generate a stable run identifier used in exported filenames."""
    if logger is not None and getattr(logger, "log_file", None):
        return Path(logger.log_file).stem
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"{prefix}_{timestamp}"


@dataclass(frozen=True)
class RunContext:
    """Stable metadata for one simulation run."""

    output_dir: Path
    run_id: str
    run_timestamp_utc: str
    git_commit: str
    model: str
    seed: int
    config_json: str

    def provenance(self) -> dict[str, str | int]:
        """Metadata attached to exported rows for sweep traceability."""
        return {
            "run_id": self.run_id,
            "run_timestamp_utc": self.run_timestamp_utc,
            "git_commit": self.git_commit,
            "model": self.model,
            "seed": self.seed,
        }


def make_run_context(
    output_dir: Path,
    logger: Any | None,
    model: str,
    seed: int,
    config_dict: dict[str, Any],
) -> RunContext:
    """Create run metadata needed by metrics/export paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        output_dir=output_dir,
        run_id=build_run_id(logger=logger),
        run_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        model=model,
        seed=seed,
        config_json=json.dumps(config_dict, sort_keys=True),
    )


def build_round_metric_rows(round_data: list[dict], run: RunContext) -> list[dict]:
    """Compute per-round aggregate metrics."""
    rows: list[dict] = []
    base = run.provenance()

    for rd in round_data:
        results = rd["results"]
        num_agents = len(results)
        num_liars = sum(1 for r in results if r["is_liar"])
        num_reported_stag = sum(1 for r in results if r["reported_is_stag"])
        num_reported_hare = num_agents - num_reported_stag
        num_flipped = sum(1 for r in results if r["was_flipped"])
        num_correct = sum(1 for r in results if r["is_correct"])
        liar_rows = [r for r in results if r["is_liar"]]
        liar_correct = sum(1 for r in liar_rows if r["is_correct"])
        confidences = [r["confidence"] for r in results]
        payoffs = [r["realized_payoff"] for r in results]

        entropy = 0.0
        for count in (num_reported_stag, num_reported_hare):
            if num_agents == 0:
                continue
            p = count / num_agents
            if p > 0:
                entropy -= p * math.log2(p)

        rows.append(
            {
                **base,
                "round": rd["round"],
                "true_action": rd["true_action"],
                "true_is_stag": rd["true_is_stag"],
                "stag_success_threshold": rd["stag_success_threshold"],
                "num_stag_reported": rd["num_stag_reported"],
                "stag_success": rd["stag_success"],
                "num_agents": num_agents,
                "num_liars": num_liars,
                "liar_share": (num_liars / num_agents) if num_agents else 0.0,
                "num_reported_stag": num_reported_stag,
                "num_reported_hare": num_reported_hare,
                "consensus_rate": (
                    max(num_reported_stag, num_reported_hare) / num_agents
                    if num_agents
                    else 0.0
                ),
                "report_entropy": entropy,
                "num_flipped": num_flipped,
                "flip_rate": (num_flipped / num_agents) if num_agents else 0.0,
                "round_accuracy": (num_correct / num_agents) if num_agents else 0.0,
                "liar_accuracy": (liar_correct / len(liar_rows)) if liar_rows else 0.0,
                "confidence_mean": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "confidence_std": pstdev(confidences) if len(confidences) > 1 else 0.0,
                "payoff_mean": sum(payoffs) / len(payoffs) if payoffs else 0.0,
                "payoff_std": pstdev(payoffs) if len(payoffs) > 1 else 0.0,
                "payoff_total": sum(payoffs),
            }
        )

    return rows


def _role_name(role: Any) -> str:
    return str(getattr(role, "name"))


def _role_is_liar(role: Any) -> bool:
    return bool(getattr(role, "is_liar"))


def build_agent_summary_rows(
    round_data: list[dict], roles: Sequence[Any], run: RunContext
) -> list[dict]:
    """Compute per-agent metrics across all rounds."""
    by_agent: dict[str, list[dict]] = {_role_name(role): [] for role in roles}
    base = run.provenance()

    for rd in round_data:
        results = rd["results"]
        for idx, result in enumerate(results):
            later_results = results[idx + 1 :]
            influence_on_later = None
            if later_results:
                matches = sum(
                    1
                    for later in later_results
                    if later["reported_is_stag"] == result["reported_is_stag"]
                )
                influence_on_later = matches / len(later_results)

            by_agent[result["agent"]].append(
                {
                    "round": rd["round"],
                    "turn_index": result["turn_index"],
                    "reported_is_stag": result["reported_is_stag"],
                    "is_correct": int(result["is_correct"]),
                    "confidence": result["confidence"],
                    "realized_payoff": result["realized_payoff"],
                    "influence_on_later": influence_on_later,
                }
            )

    rows: list[dict] = []
    for role in roles:
        agent_name = _role_name(role)
        agent_model = str(getattr(role, "model", run.model))
        decisions = sorted(by_agent[agent_name], key=lambda x: x["round"])
        confidences = [d["confidence"] for d in decisions]
        correctness = [d["is_correct"] for d in decisions]

        calibration_errors = [abs(d["confidence"] - d["is_correct"]) for d in decisions]
        persistence_rate = None
        if len(decisions) >= 2:
            unchanged = sum(
                1
                for i in range(1, len(decisions))
                if decisions[i]["reported_is_stag"] == decisions[i - 1]["reported_is_stag"]
            )
            persistence_rate = unchanged / (len(decisions) - 1)

        influence_values = [
            d["influence_on_later"] for d in decisions if d["influence_on_later"] is not None
        ]

        rows.append(
            {
                **base,
                "agent": agent_name,
                "agent_model": agent_model,
                "is_liar": _role_is_liar(role),
                "num_rounds_seen": len(decisions),
                "accuracy": (sum(correctness) / len(correctness)) if correctness else 0.0,
                "confidence_mean": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "confidence_std": pstdev(confidences) if len(confidences) > 1 else 0.0,
                "payoff_mean": (
                    sum(d["realized_payoff"] for d in decisions) / len(decisions)
                    if decisions
                    else 0.0
                ),
                "payoff_std": (
                    pstdev([d["realized_payoff"] for d in decisions])
                    if len(decisions) > 1
                    else 0.0
                ),
                "calibration_error": (
                    sum(calibration_errors) / len(calibration_errors)
                    if calibration_errors
                    else 0.0
                ),
                "persistence_rate": persistence_rate,
                "influence_rate": (
                    sum(influence_values) / len(influence_values)
                    if influence_values
                    else None
                ),
            }
        )

    return rows


def _write_runs_index_row(
    run: RunContext,
    num_agents: int,
    num_rounds: int,
    lie_fraction: float,
    order_ablation: str,
    adversary_ablation: str,
    heterogeneity_ablation: str,
    h3_liar_policy: str,
    model_pool: str,
    stag_success_threshold: int,
    payoff_stag_success: float,
    payoff_hare_when_stag_success: float,
    payoff_stag_fail: float,
    payoff_hare_fail: float,
    accuracy: float,
    liar_accuracy: float,
) -> str:
    """Append one metadata row per run to a sweep-friendly runs index CSV."""
    runs_index_csv = run.output_dir / "stag_hunt_runs.csv"
    write_header = not runs_index_csv.exists()
    with runs_index_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "run_timestamp_utc",
                "git_commit",
                "model",
                "seed",
                "num_agents",
                "num_rounds",
                "lie_fraction",
                "order_ablation",
                "adversary_ablation",
                "heterogeneity_ablation",
                "h3_liar_policy",
                "model_pool",
                "stag_success_threshold",
                "payoff_stag_success",
                "payoff_hare_when_stag_success",
                "payoff_stag_fail",
                "payoff_hare_fail",
                "accuracy",
                "liar_accuracy",
                "config_json",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                **run.provenance(),
                "num_agents": num_agents,
                "num_rounds": num_rounds,
                "lie_fraction": lie_fraction,
                "order_ablation": order_ablation,
                "adversary_ablation": adversary_ablation,
                "heterogeneity_ablation": heterogeneity_ablation,
                "h3_liar_policy": h3_liar_policy,
                "model_pool": model_pool,
                "stag_success_threshold": stag_success_threshold,
                "payoff_stag_success": payoff_stag_success,
                "payoff_hare_when_stag_success": payoff_hare_when_stag_success,
                "payoff_stag_fail": payoff_stag_fail,
                "payoff_hare_fail": payoff_hare_fail,
                "accuracy": accuracy,
                "liar_accuracy": liar_accuracy,
                "config_json": run.config_json,
            }
        )
    return str(runs_index_csv)


def export_csv_results(
    round_data: list[dict],
    run: RunContext,
    round_metrics_rows: list[dict],
    agent_summary_rows: list[dict],
    num_agents: int,
    num_rounds: int,
    lie_fraction: float,
    order_ablation: str,
    adversary_ablation: str,
    heterogeneity_ablation: str,
    h3_liar_policy: str,
    model_pool: str,
    stag_success_threshold: int,
    payoff_stag_success: float,
    payoff_hare_when_stag_success: float,
    payoff_stag_fail: float,
    payoff_hare_fail: float,
    accuracy: float,
    liar_accuracy: float,
) -> dict[str, str]:
    """Persist simulation outputs in CSV format for pandas workflows."""
    round_csv = run.output_dir / f"{run.run_id}_round_metrics.csv"
    agent_metrics_csv = run.output_dir / f"{run.run_id}_agent_metrics.csv"
    agent_text_csv = run.output_dir / f"{run.run_id}_agent_text.csv"
    agent_summary_csv = run.output_dir / f"{run.run_id}_agent_summary.csv"

    with round_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "run_timestamp_utc",
                "git_commit",
                "model",
                "seed",
                "round",
                "true_action",
                "true_is_stag",
                "stag_success_threshold",
                "num_stag_reported",
                "stag_success",
                "num_agents",
                "num_liars",
                "liar_share",
                "num_reported_stag",
                "num_reported_hare",
                "consensus_rate",
                "report_entropy",
                "num_flipped",
                "flip_rate",
                "round_accuracy",
                "liar_accuracy",
                "confidence_mean",
                "confidence_std",
                "payoff_mean",
                "payoff_std",
                "payoff_total",
            ],
        )
        writer.writeheader()
        writer.writerows(round_metrics_rows)

    with agent_metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "run_timestamp_utc",
                "git_commit",
                "model",
                "seed",
                "round",
                "turn_index",
                "agent",
                "agent_model",
                "is_liar",
                "true_action",
                "true_is_stag",
                "original_action",
                "original_is_stag",
                "reported_action",
                "reported_is_stag",
                "was_flipped",
                "is_correct",
                "confidence",
                "realized_payoff",
                "stag_success",
                "influence_on_later_agents",
            ],
        )
        writer.writeheader()
        base = run.provenance()
        for rd in round_data:
            results = rd["results"]
            for idx, result in enumerate(results):
                later_results = results[idx + 1 :]
                influence_on_later = None
                if later_results:
                    matches = sum(
                        1
                        for later in later_results
                        if later["reported_is_stag"] == result["reported_is_stag"]
                    )
                    influence_on_later = matches / len(later_results)

                writer.writerow(
                    {
                        **base,
                        "round": rd["round"],
                        "turn_index": result["turn_index"],
                        "agent": result["agent"],
                        "agent_model": result.get("agent_model", run.model),
                        "is_liar": result["is_liar"],
                        "true_action": result["true_action"],
                        "true_is_stag": result["true_is_stag"],
                        "original_action": result["original_action"],
                        "original_is_stag": result["original_is_stag"],
                        "reported_action": result["reported_action"],
                        "reported_is_stag": result["reported_is_stag"],
                        "was_flipped": result["was_flipped"],
                        "is_correct": result["is_correct"],
                        "confidence": result["confidence"],
                        "realized_payoff": result["realized_payoff"],
                        "stag_success": result["stag_success"],
                        "influence_on_later_agents": influence_on_later,
                    }
                )

    with agent_text_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "run_timestamp_utc",
                "git_commit",
                "model",
                "seed",
                "round",
                "turn_index",
                "agent",
                "justification",
            ],
        )
        writer.writeheader()
        base = run.provenance()
        for rd in round_data:
            for result in rd["results"]:
                writer.writerow(
                    {
                        **base,
                        "round": rd["round"],
                        "turn_index": result["turn_index"],
                        "agent": result["agent"],
                        "justification": result["justification"],
                    }
                )

    with agent_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "run_timestamp_utc",
                "git_commit",
                "model",
                "seed",
                "agent",
                "agent_model",
                "is_liar",
                "num_rounds_seen",
                "accuracy",
                "confidence_mean",
                "confidence_std",
                "payoff_mean",
                "payoff_std",
                "calibration_error",
                "persistence_rate",
                "influence_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(agent_summary_rows)

    runs_index_csv = _write_runs_index_row(
        run=run,
        num_agents=num_agents,
        num_rounds=num_rounds,
        lie_fraction=lie_fraction,
        order_ablation=order_ablation,
        adversary_ablation=adversary_ablation,
        heterogeneity_ablation=heterogeneity_ablation,
        h3_liar_policy=h3_liar_policy,
        model_pool=model_pool,
        stag_success_threshold=stag_success_threshold,
        payoff_stag_success=payoff_stag_success,
        payoff_hare_when_stag_success=payoff_hare_when_stag_success,
        payoff_stag_fail=payoff_stag_fail,
        payoff_hare_fail=payoff_hare_fail,
        accuracy=accuracy,
        liar_accuracy=liar_accuracy,
    )

    return {
        "round_metrics": str(round_csv),
        "agent_metrics": str(agent_metrics_csv),
        "agent_text": str(agent_text_csv),
        "agent_summary": str(agent_summary_csv),
        "runs_index": runs_index_csv,
    }
