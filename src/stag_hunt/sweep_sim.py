from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import io
import random
import secrets
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from flashlite import Flashlite, InspectLogger, RateLimitConfig
from flashlite.observability import convert_flashlite_logs_to_inspect

from stag_hunt.results import SharedCSVPaths
from stag_hunt.sim import PROMPTS_DIR, GameConfig, StagHuntSimulation

BASE_ORDER_ABLATION = "a1"
BASE_ADVERSARY_ABLATION = "base"
BASE_HETEROGENEITY_ABLATION = "h1"
BASE_ABLATION_CODE = "base"


@dataclass(frozen=True)
class SweepPoint:
    """One configuration point in the sweep grid."""

    model: str
    num_agents: int
    num_rounds: int
    num_liars: int
    stag_success_threshold: int
    payoff_stag_success: float
    payoff_hare_when_stag_success: float
    payoff_stag_fail: float
    payoff_hare_fail: float
    order_ablation: str
    adversary_ablation: str
    heterogeneity_ablation: str
    h3_liar_policy: str
    model_pool: tuple[str, ...]
    ablation_code: str
    replicate: int
    seed: int
    point_id: str = ""


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in _parse_csv_values(raw)]


def _parse_float_list(raw: str) -> list[float]:
    return [float(v) for v in _parse_csv_values(raw)]


def _parse_agent_configs(raw: str) -> list[tuple[int, int]]:
    """Parse 'N,L;N,L;...' agent config pairs (num_agents, num_liars)."""
    configs: list[tuple[int, int]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid agent config '{entry}': expected 'num_agents,num_liars'"
            )
        configs.append((int(parts[0]), int(parts[1])))
    return configs


def _parse_payoff_tuples(raw: str) -> list[tuple[float, float, float, float]]:
    """Parse 'R,T,S,P;R,T,S,P;...' payoff tuples."""
    tuples: list[tuple[float, float, float, float]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid payoff tuple '{entry}': expected "
                "'stag_success,hare_when_stag_success,stag_fail,hare_fail'"
            )
        tuples.append(
            (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        )
    return tuples


def _normalize_model_pool(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(_parse_csv_values(value))
    if isinstance(value, (list, tuple)):
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(normalized)
    return ()


def _model_pool_to_csv(values: tuple[str, ...]) -> str:
    return ",".join(values)


COMPLETED_POINTS_FILE = "stag_hunt_completed.csv"


def _generate_sweep_id(prefix: str) -> str:
    """Generate a unique sweep ID with prefix and timestamp (computed once per sweep)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add short random suffix for uniqueness if multiple sweeps start same second
    suffix = secrets.token_hex(4)
    return f"{prefix}_{timestamp}_{suffix}"


def _compute_point_id(point: SweepPoint) -> str:
    """Deterministic short hash from all config fields (excluding point_id itself)."""
    key = (
        point.model,
        point.num_agents,
        point.num_rounds,
        point.num_liars,
        point.stag_success_threshold,
        point.payoff_stag_success,
        point.payoff_hare_when_stag_success,
        point.payoff_stag_fail,
        point.payoff_hare_fail,
        point.order_ablation,
        point.adversary_ablation,
        point.heterogeneity_ablation,
        point.h3_liar_policy,
        point.model_pool,
        point.ablation_code,
        point.replicate,
        point.seed,
    )
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]


def _assign_point_ids(points: list[SweepPoint]) -> list[SweepPoint]:
    """Assign a deterministic point_id to every point that lacks one."""
    return [
        replace(p, point_id=_compute_point_id(p)) if not p.point_id else p
        for p in points
    ]


def _load_completed_ids(log_dir: Path) -> set[str]:
    """Load point_ids of previously completed runs from the tracking file."""
    path = log_dir / COMPLETED_POINTS_FILE
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(row["point_id"])
    return completed


def _mark_completed(log_dir: Path, point_id: str, status: str) -> None:
    """Append one row to the tracking file after a run finishes."""
    path = log_dir / COMPLETED_POINTS_FILE
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "status", "timestamp"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "point_id": point_id,
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Stag Hunt simulation configs.")
    parser.add_argument("--models", type=str, default="openai/gpt-5-nano")
    parser.add_argument(
        "--agent-configs",
        type=str,
        default="4,1",
        help="Semicolon-separated 'num_agents,num_liars' pairs, e.g. '4,1;6,2'",
    )
    parser.add_argument("--num-rounds", type=str, default="2")
    parser.add_argument("--stag-thresholds", type=str, default="")
    parser.add_argument(
        "--payoffs",
        type=str,
        default="4.0,2.0,0.0,2.0",
        help=(
            "Semicolon-separated payoff tuples "
            "'stag_success,hare_when_stag_success,stag_fail,hare_fail', "
            "e.g. '4,2,0,2;3,1.5,0,1'"
        ),
    )
    parser.add_argument("--ablations", type=str, default="")
    parser.add_argument("--ablation-subset-runs", type=int, default=12)
    parser.add_argument("--model-pool", type=str, default="")
    parser.add_argument(
        "--h3-liar-policy",
        type=str,
        choices=["strongest_liars", "weakest_liars"],
        default="strongest_liars",
    )
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--sweep-points-file", type=str, default="")
    parser.add_argument("--save-sweep-points", type=str, default="")
    parser.add_argument("--output-tokens-per-turn", type=int, default=2000)
    parser.add_argument("--input-overhead-tokens-per-turn", type=int, default=300)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--tpm", type=int, default=20000)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--eval-prefix", type=str, default="stag_hunt_simulation")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default="",
        help=(
            "Unique ID for this sweep run. If not provided, one is auto-generated. "
            "Reuse an existing sweep-id to resume a previous run."
        ),
    )
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _point_from_mapping(data: dict[str, Any]) -> SweepPoint:
    num_agents = int(data["num_agents"])
    default_threshold = num_agents
    return SweepPoint(
        model=str(data["model"]),
        num_agents=num_agents,
        num_rounds=int(data["num_rounds"]),
        num_liars=int(data["num_liars"]),
        stag_success_threshold=int(
            data.get("stag_success_threshold", default_threshold)
        ),
        payoff_stag_success=float(data.get("payoff_stag_success", 4.0)),
        payoff_hare_when_stag_success=float(
            data.get("payoff_hare_when_stag_success", 3.0)
        ),
        payoff_stag_fail=float(data.get("payoff_stag_fail", 0.0)),
        payoff_hare_fail=float(data.get("payoff_hare_fail", 2.0)),
        order_ablation=str(data.get("order_ablation", BASE_ORDER_ABLATION)),
        adversary_ablation=str(data.get("adversary_ablation", BASE_ADVERSARY_ABLATION)),
        heterogeneity_ablation=str(
            data.get("heterogeneity_ablation", BASE_HETEROGENEITY_ABLATION)
        ),
        h3_liar_policy=str(data.get("h3_liar_policy", "strongest_liars")),
        model_pool=_normalize_model_pool(data.get("model_pool")),
        ablation_code=str(data.get("ablation_code", BASE_ABLATION_CODE)),
        replicate=int(data.get("replicate", 0)),
        seed=int(data["seed"]),
        point_id=str(data.get("point_id", "")),
    )


def _load_sweep_points(path: Path) -> list[SweepPoint]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        points = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(_point_from_mapping(row))
        return points
    raise ValueError("Unsupported sweep point file type; use .csv")


def _build_sweep_points_from_grid(args: argparse.Namespace) -> list[SweepPoint]:
    models = _parse_csv_values(args.models)
    agent_configs = _parse_agent_configs(args.agent_configs)
    num_rounds = _parse_int_list(args.num_rounds)
    payoff_tuples = _parse_payoff_tuples(args.payoffs)
    configured_stag_thresholds = (
        _parse_int_list(args.stag_thresholds) if args.stag_thresholds else None
    )

    points: list[SweepPoint] = []
    base_model_pool = ()
    seed_counter = args.seed_start
    for (
        model,
        (agents, num_liars),
        rounds,
        (
            payoff_stag_success,
            payoff_hare_when_stag_success,
            payoff_stag_fail,
            payoff_hare_fail,
        ),
    ) in product(
        models,
        agent_configs,
        num_rounds,
        payoff_tuples,
    ):
        stag_thresholds = configured_stag_thresholds or [agents]
        stag_thresholds = [t for t in stag_thresholds if 1 <= t <= agents]
        if not stag_thresholds:
            continue

        for stag_success_threshold in stag_thresholds:
            for replicate in range(args.replicates):
                points.append(
                    SweepPoint(
                        model=model,
                        num_agents=agents,
                        num_rounds=rounds,
                        num_liars=num_liars,
                        stag_success_threshold=stag_success_threshold,
                        payoff_stag_success=payoff_stag_success,
                        payoff_hare_when_stag_success=payoff_hare_when_stag_success,
                        payoff_stag_fail=payoff_stag_fail,
                        payoff_hare_fail=payoff_hare_fail,
                        order_ablation=BASE_ORDER_ABLATION,
                        adversary_ablation=BASE_ADVERSARY_ABLATION,
                        heterogeneity_ablation=BASE_HETEROGENEITY_ABLATION,
                        h3_liar_policy=args.h3_liar_policy,
                        model_pool=base_model_pool,
                        ablation_code=BASE_ABLATION_CODE,
                        replicate=replicate,
                        seed=seed_counter,
                    )
                )
                seed_counter += 1

    return points


def _build_sweep_points(args: argparse.Namespace) -> list[SweepPoint]:
    if args.sweep_points_file:
        points = _load_sweep_points(Path(args.sweep_points_file))
    else:
        points = _build_sweep_points_from_grid(args)

    if args.max_runs > 0:
        points = points[: args.max_runs]

    ablation_codes = [c.strip().lower() for c in _parse_csv_values(args.ablations)]
    if not ablation_codes:
        return points

    allowed_codes = {"b3", "a1", "a2", "a3", "h1", "h2", "h3"}
    invalid_codes = [c for c in ablation_codes if c not in allowed_codes]
    if invalid_codes:
        raise ValueError(
            f"Unsupported ablation codes: {', '.join(invalid_codes)}. "
            f"Allowed: {', '.join(sorted(allowed_codes))}"
        )

    # Sample a subset of base points to apply ablation variants to.
    subset_size = max(1, args.ablation_subset_runs)
    grouped_points: dict[tuple[Any, ...], list[SweepPoint]] = defaultdict(list)
    for point in points:
        # Group by parameterization and keep all replicate/seed variants together.
        key = (
            point.model,
            point.num_agents,
            point.num_rounds,
            point.num_liars,
            point.stag_success_threshold,
            point.payoff_stag_success,
            point.payoff_hare_when_stag_success,
            point.payoff_stag_fail,
            point.payoff_hare_fail,
            point.order_ablation,
            point.adversary_ablation,
            point.heterogeneity_ablation,
            point.h3_liar_policy,
            point.model_pool,
        )
        grouped_points[key].append(point)

    unique_keys = list(grouped_points.keys())
    subset_size = min(subset_size, len(unique_keys))
    subset_rng = random.Random(args.seed_start)
    sampled_keys = subset_rng.sample(unique_keys, k=subset_size)
    # One representative point per parameterization (first replicate) for ablations.
    ablation_bases = [grouped_points[key][0] for key in sampled_keys]

    model_pool_text = args.model_pool.strip() or args.models
    model_pool = tuple(_parse_csv_values(model_pool_text))

    # Start with ALL original base points, then append ablation variants of the subset.
    ablation_points: list[SweepPoint] = list(points)
    for code in ablation_codes:
        for point in ablation_bases:
            variant = replace(point, ablation_code=code)
            if code == "b3":
                variant = replace(variant, adversary_ablation="b3")
            elif code in {"a1", "a2", "a3"}:
                variant = replace(variant, order_ablation=code)
            elif code == "h1":
                variant = replace(variant, heterogeneity_ablation="h1", model_pool=())
            elif code == "h2":
                variant = replace(
                    variant,
                    heterogeneity_ablation="h2",
                    model_pool=model_pool,
                )
            elif code == "h3":
                variant = replace(
                    variant,
                    heterogeneity_ablation="h3",
                    h3_liar_policy=args.h3_liar_policy,
                    model_pool=model_pool,
                )
            ablation_points.append(variant)

    return ablation_points


def _filter_valid_sweep_points(
    points: list[SweepPoint],
) -> tuple[list[SweepPoint], list[str]]:
    """Keep valid sweep points and collect warnings for invalid ones."""
    warnings: list[str] = []
    valid_points: list[SweepPoint] = []
    for idx, point in enumerate(points, start=1):
        model_pool = list(point.model_pool) if point.model_pool else None
        try:
            GameConfig(
                model=point.model,
                num_agents=point.num_agents,
                num_rounds=point.num_rounds,
                num_liars=point.num_liars,
                stag_success_threshold=point.stag_success_threshold,
                payoff_stag_success=point.payoff_stag_success,
                payoff_hare_when_stag_success=point.payoff_hare_when_stag_success,
                payoff_stag_fail=point.payoff_stag_fail,
                payoff_hare_fail=point.payoff_hare_fail,
                order_ablation=point.order_ablation,  # a1/a2/a3
                adversary_ablation=point.adversary_ablation,  # base/b3
                heterogeneity_ablation=point.heterogeneity_ablation,  # h1/h2/h3
                h3_liar_policy=point.h3_liar_policy,
                model_pool=model_pool,
                seed=point.seed,
            )
            valid_points.append(point)
        except ValueError as exc:
            warnings.append(
                f"point #{idx} model={point.model} "
                f"N={point.num_agents} rounds={point.num_rounds} "
                f"num_liars={point.num_liars} M={point.stag_success_threshold} "
                f"payoffs=[R={point.payoff_stag_success}, "
                f"T={point.payoff_hare_when_stag_success}, "
                f"S={point.payoff_stag_fail}, P={point.payoff_hare_fail}] "
                f"ablation={point.ablation_code}: {exc}"
            )
    return valid_points, warnings


def _build_sweep_summary_path(log_dir: Path, sweep_id: str) -> Path:
    """Build path for sweep summary CSV using the sweep_id."""
    return log_dir / f"{sweep_id}_sweep_summary.csv"


def _build_sweep_points_path(log_dir: Path, sweep_id: str, configured: str) -> Path:
    """Build path for sweep points CSV using the sweep_id."""
    if configured:
        return Path(configured)
    return log_dir / f"{sweep_id}_sweep_points.csv"


def _save_sweep_points(path: Path, points: list[SweepPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {"", ".csv"}:
        fieldnames = [
            "point_id",
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
            "heterogeneity_ablation",
            "h3_liar_policy",
            "model_pool",
            "ablation_code",
            "replicate",
            "seed",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for point in points:
                row = asdict(point)
                row["model_pool"] = _model_pool_to_csv(point.model_pool)
                writer.writerow(row)
        return
    raise ValueError(
        f"Unsupported sweep point output format '{suffix or '<none>'}'. Use .csv."
    )


def _estimate_tokens_for_point(
    point: SweepPoint,
    output_tokens_per_turn: int,
    input_overhead_tokens_per_turn: int,
) -> dict[str, int]:
    """Rough token estimate.

    Assumptions:
    - One agent-turn per (agent, round): T = num_agents * num_rounds.
    - Output tokens are ~constant per turn.
    - Input includes fixed overhead per turn plus replayed prior outputs.
      Replay term grows with history: output_per_turn * sum_{t=0..T-1} t.
    """
    turns = point.num_agents * point.num_rounds
    est_output = turns * output_tokens_per_turn
    est_input_overhead = turns * input_overhead_tokens_per_turn
    est_input_history = output_tokens_per_turn * (turns * (turns - 1) // 2)
    est_input = est_input_overhead + est_input_history
    return {
        "estimated_turns": turns,
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_total_tokens": est_input + est_output,
    }


def _write_sweep_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows_to_write = sorted(rows, key=lambda r: int(r.get("run_index", 0)))
    fieldnames = list(rows_to_write[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)


async def _run_single(
    point: SweepPoint,
    client: Flashlite,
    args: argparse.Namespace,
    run_index: int,
    run_total: int,
    sweep_id: str,
    shared_logger: InspectLogger,
    shared_csv_paths: SharedCSVPaths,
) -> dict[str, Any]:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    estimates = _estimate_tokens_for_point(
        point=point,
        output_tokens_per_turn=args.output_tokens_per_turn,
        input_overhead_tokens_per_turn=args.input_overhead_tokens_per_turn,
    )
    model_pool = list(point.model_pool) if point.model_pool else None

    config = GameConfig(
        model=point.model,
        num_agents=point.num_agents,
        num_rounds=point.num_rounds,
        num_liars=point.num_liars,
        stag_success_threshold=point.stag_success_threshold,
        payoff_stag_success=point.payoff_stag_success,
        payoff_hare_when_stag_success=point.payoff_hare_when_stag_success,
        payoff_stag_fail=point.payoff_stag_fail,
        payoff_hare_fail=point.payoff_hare_fail,
        order_ablation=point.order_ablation,  # a1/a2/a3
        adversary_ablation=point.adversary_ablation,  # base/b3
        heterogeneity_ablation=point.heterogeneity_ablation,  # h1/h2/h3
        h3_liar_policy=point.h3_liar_policy,
        model_pool=model_pool,
        seed=point.seed,
    )

    # Use the shared logger for the entire sweep (one JSONL file)
    logger = shared_logger

    print(
        f"[{run_index}/{run_total}] {point.point_id} "
        f"model={point.model} agents={point.num_agents} rounds={point.num_rounds} "
        f"num_liars={point.num_liars} M={point.stag_success_threshold} "
        f"ablation={point.ablation_code} "
        f"replicate={point.replicate} seed={point.seed} "
        f"est_tokens≈{estimates['estimated_total_tokens']}"
    )

    started = time.perf_counter()
    status = "ok"
    error = ""
    results: dict[str, Any] = {}
    cost_cumulative_before = client.total_cost
    # Use consistent run_id based on sweep_id and point_id
    run_id = f"{sweep_id}_{point.point_id}"
    try:
        simulation = StagHuntSimulation(
            client=client,
            config=config,
            logger=logger,
            run_id_override=run_id,
            shared_csv_paths=shared_csv_paths,
        )
        if args.verbose:
            results = await simulation.run_game()
        else:
            if args.max_concurrency == 1:
                with contextlib.redirect_stdout(io.StringIO()):
                    results = await simulation.run_game()
            else:
                results = await simulation.run_game()

    except Exception as exc:  # pragma: no cover - safety path
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        if args.fail_fast:
            raise
    # Note: Don't close logger here - it's shared across all runs

    elapsed_sec = time.perf_counter() - started
    csv_exports = results.get("csv_exports", {})
    return {
        "point_id": point.point_id,
        "status": status,
        "error": error,
        "model": point.model,
        "num_agents": point.num_agents,
        "num_rounds": point.num_rounds,
        "num_liars": point.num_liars,
        "stag_success_threshold": point.stag_success_threshold,
        "payoff_stag_success": point.payoff_stag_success,
        "payoff_hare_when_stag_success": point.payoff_hare_when_stag_success,
        "payoff_stag_fail": point.payoff_stag_fail,
        "payoff_hare_fail": point.payoff_hare_fail,
        "order_ablation": point.order_ablation,
        "adversary_ablation": point.adversary_ablation,
        "heterogeneity_ablation": point.heterogeneity_ablation,
        "h3_liar_policy": point.h3_liar_policy,
        "model_pool": _model_pool_to_csv(point.model_pool),
        "ablation_code": point.ablation_code,
        "replicate": point.replicate,
        "seed": point.seed,
        **estimates,
        "accuracy": results.get("accuracy"),
        "liar_accuracy": results.get("liar_accuracy"),
        "total_decisions": results.get("total_decisions"),
        "total_tokens": (results.get("chat_stats") or {}).get("total_tokens"),
        "cost_usd_cumulative_before": cost_cumulative_before,
        "cost_usd_cumulative_after": client.total_cost,
        "elapsed_sec": elapsed_sec,
        "inspect_log": getattr(logger, "log_file", ""),
        "round_metrics_csv": csv_exports.get("round_metrics", ""),
        "agent_metrics_csv": csv_exports.get("agent_metrics", ""),
        "agent_text_csv": csv_exports.get("agent_text", ""),
        "agent_summary_csv": csv_exports.get("agent_summary", ""),
        "runs_index_csv": csv_exports.get("runs_index", ""),
    }


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    points = _build_sweep_points(args)
    if not points:
        print("No sweep points generated.")
        return
    initial_count = len(points)
    points, dropped_warnings = _filter_valid_sweep_points(points)
    dropped_count = len(dropped_warnings)
    if dropped_count:
        print(
            f"Warning: dropped {dropped_count} invalid sweep points "
            f"(kept {len(points)} / {initial_count})."
        )
        for warning in dropped_warnings[:5]:
            print(f"  - {warning}")
        if dropped_count > 5:
            print(f"  ... and {dropped_count - 5} more")
    if not points:
        print("No valid sweep points remain after filtering. Nothing to run.")
        return

    # Assign deterministic IDs and check for prior completions.
    points = _assign_point_ids(points)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate or reuse sweep_id for consistent file naming
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using provided sweep-id: {sweep_id}")
    else:
        sweep_id = _generate_sweep_id(args.eval_prefix)
        print(f"Generated sweep-id: {sweep_id}")

    completed_ids = _load_completed_ids(log_dir)
    total_generated = len(points)
    if completed_ids:
        points = [p for p in points if p.point_id not in completed_ids]
        skipped = total_generated - len(points)
        if skipped:
            print(
                f"Resuming: skipped {skipped} already-completed points "
                f"({len(points)} remaining of {total_generated})"
            )
    if not points:
        print("All sweep points already completed. Nothing to run.")
        return

    summary_path = _build_sweep_summary_path(log_dir, sweep_id)
    points_path = _build_sweep_points_path(log_dir, sweep_id, args.save_sweep_points)
    _save_sweep_points(points_path, points)

    total_est_input = 0
    total_est_output = 0
    total_est_all = 0
    for point in points:
        est = _estimate_tokens_for_point(
            point=point,
            output_tokens_per_turn=args.output_tokens_per_turn,
            input_overhead_tokens_per_turn=args.input_overhead_tokens_per_turn,
        )
        total_est_input += est["estimated_input_tokens"]
        total_est_output += est["estimated_output_tokens"]
        total_est_all += est["estimated_total_tokens"]

    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep runs: {len(points)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Sweep points: {points_path}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"To resume this sweep later, use: --sweep-id {sweep_id}")
    print(
        "Estimated tokens (all runs): "
        f"input≈{total_est_input:,}, output≈{total_est_output:,}, total≈{total_est_all:,}"
    )
    if args.ablations:
        print(
            "Ablation mode enabled: "
            f"{args.ablations} on up to {max(1, args.ablation_subset_runs)} base parameter points"
        )
    if not args.skip_confirmation:
        answer = input("Proceed with sweep run? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Sweep cancelled.")
            return

    shared_client = Flashlite(
        default_model=points[0].model,
        template_dir=PROMPTS_DIR,
        track_costs=True,
        rate_limit=RateLimitConfig(
            requests_per_minute=args.rpm,
            tokens_per_minute=args.tpm,
        ),
    )

    # Create ONE shared logger for all runs - all entries go to a single JSONL file
    shared_logger = InspectLogger(
        log_dir=str(log_dir),
        eval_id=sweep_id,  # Use sweep_id directly as the eval_id (no extra timestamp)
        append=True,
    )

    # Create shared CSV paths - all runs append to these 4 files
    shared_csv_paths = SharedCSVPaths(
        round_metrics=log_dir / f"{sweep_id}_round_metrics.csv",
        agent_metrics=log_dir / f"{sweep_id}_agent_metrics.csv",
        agent_text=log_dir / f"{sweep_id}_agent_text.csv",
        agent_summary=log_dir / f"{sweep_id}_agent_summary.csv",
    )

    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def _run_with_limit(
        run_index: int, point: SweepPoint
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            row = await _run_single(
                point=point,
                client=shared_client,
                args=args,
                run_index=run_index,
                run_total=len(points),
                sweep_id=sweep_id,
                shared_logger=shared_logger,
                shared_csv_paths=shared_csv_paths,
            )
            return run_index, row

    rows: list[dict[str, Any]] = []
    tasks = [
        asyncio.create_task(_run_with_limit(idx, point))
        for idx, point in enumerate(points, start=1)
    ]
    for completed in asyncio.as_completed(tasks):
        run_index, row = await completed
        row["run_index"] = run_index
        rows.append(row)
        _write_sweep_summary(summary_path, rows)
        _mark_completed(log_dir, row["point_id"], row["status"])

        if row["status"] == "ok":
            acc = row.get("accuracy")
            liar_acc = row.get("liar_accuracy")
            acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "n/a"
            liar_acc_str = (
                f"{liar_acc:.3f}" if isinstance(liar_acc, (int, float)) else "n/a"
            )
            print(
                f"  -> accuracy={acc_str} liar_accuracy={liar_acc_str} "
                f"shared_cost_total=${row['cost_usd_cumulative_after']:.4f} "
                f"time={row['elapsed_sec']:.1f}s"
            )
        else:
            print(f"  -> FAILED: {row['error']}")

    # Close the shared logger and convert to Inspect format
    if getattr(shared_logger, "_log_file", None):
        convert_flashlite_logs_to_inspect(shared_logger._log_file)
    shared_logger.close()

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    total_elapsed = sum(float(row["elapsed_sec"]) for row in rows)
    print(f"\nCompleted sweep: {ok_count}/{len(rows)} successful")
    print(f"Sweep ID: {sweep_id}")
    print(f"Shared client total cost: ${shared_client.total_cost:.4f}")
    print(f"Sum of per-run wall times: {total_elapsed:.1f}s")
    print("\nOutput files:")
    print(f"  Sweep summary: {summary_path}")
    print(f"  JSONL log: {shared_logger.log_file}")
    print(f"  Round metrics: {shared_csv_paths.round_metrics}")
    print(f"  Agent metrics: {shared_csv_paths.agent_metrics}")
    print(f"  Agent text: {shared_csv_paths.agent_text}")
    print(f"  Agent summary: {shared_csv_paths.agent_summary}")
    if ok_count < len(rows):
        print(f"\nTo resume failed points, run with: --sweep-id {sweep_id}")


if __name__ == "__main__":
    asyncio.run(main())
