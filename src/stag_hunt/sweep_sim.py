"""Parameter sweep runner for the Stag Hunt simulation.

Example grid sweep:
    uv run python -m stag_hunt.sweep_sim \
      --models openai/gpt-5-nano,openai/gpt-5-mini \
      --num-agents 4,6 \
      --num-rounds 2,4 \
      --lie-fractions 0.0,0.25 \
      --replicates 3 \
      --seed-start 100

Example explicit points sweep:
    uv run python -m stag_hunt.sweep_sim \
      --sweep-points-file ./logs/my_points.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import io
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from flashlite import Flashlite, InspectLogger, RateLimitConfig
from flashlite.observability import convert_flashlite_logs_to_inspect

from stag_hunt.sim import PROMPTS_DIR, GameConfig, StagHuntSimulation


@dataclass(frozen=True)
class SweepPoint:
    """One configuration point in the sweep grid."""

    model: str
    num_agents: int
    num_rounds: int
    lie_fraction: float
    stag_success_threshold: int
    payoff_stag_success: float
    payoff_hare_when_stag_success: float
    payoff_stag_fail: float
    payoff_hare_fail: float
    order_ablation: str
    adversary_ablation: str
    heterogeneity_ablation: str
    h3_liar_policy: str
    model_pool_csv: str
    ablation_code: str
    replicate: int
    seed: int


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in _parse_csv_values(raw)]


def _parse_float_list(raw: str) -> list[float]:
    return [float(v) for v in _parse_csv_values(raw)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Stag Hunt simulation configs.")
    parser.add_argument("--models", type=str, default="openai/gpt-5-nano")
    parser.add_argument("--num-agents", type=str, default="4")
    parser.add_argument("--num-rounds", type=str, default="2")
    parser.add_argument("--lie-fractions", type=str, default="0.25")
    parser.add_argument("--stag-thresholds", type=str, default="")
    parser.add_argument("--payoff-stag-success", type=str, default="4.0")
    parser.add_argument("--payoff-hare-when-stag-success", type=str, default="3.0")
    parser.add_argument("--payoff-stag-fail", type=str, default="0.0")
    parser.add_argument("--payoff-hare-fail", type=str, default="2.0")
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
        lie_fraction=float(data["lie_fraction"]),
        stag_success_threshold=int(data.get("stag_success_threshold", default_threshold)),
        payoff_stag_success=float(data.get("payoff_stag_success", 4.0)),
        payoff_hare_when_stag_success=float(
            data.get("payoff_hare_when_stag_success", 3.0)
        ),
        payoff_stag_fail=float(data.get("payoff_stag_fail", 0.0)),
        payoff_hare_fail=float(data.get("payoff_hare_fail", 2.0)),
        order_ablation=str(data.get("order_ablation", "a1")),
        adversary_ablation=str(data.get("adversary_ablation", "base")),
        heterogeneity_ablation=str(data.get("heterogeneity_ablation", "h1")),
        h3_liar_policy=str(data.get("h3_liar_policy", "strongest_liars")),
        model_pool_csv=str(data.get("model_pool_csv", "")),
        ablation_code=str(data.get("ablation_code", "base")),
        replicate=int(data.get("replicate", 0)),
        seed=int(data["seed"]),
    )


def _load_sweep_points(path: Path) -> list[SweepPoint]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON sweep point file must contain a list of objects")
        return [_point_from_mapping(item) for item in raw]
    if suffix == ".jsonl":
        points: list[SweepPoint] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            points.append(_point_from_mapping(json.loads(line)))
        return points
    if suffix == ".csv":
        points = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(_point_from_mapping(row))
        return points
    raise ValueError("Unsupported sweep point file type; use .json, .jsonl, or .csv")


def _build_sweep_points_from_grid(args: argparse.Namespace) -> list[SweepPoint]:
    models = _parse_csv_values(args.models)
    num_agents = _parse_int_list(args.num_agents)
    num_rounds = _parse_int_list(args.num_rounds)
    lie_fractions = _parse_float_list(args.lie_fractions)
    payoff_stag_success_values = _parse_float_list(args.payoff_stag_success)
    payoff_hare_when_stag_success_values = _parse_float_list(
        args.payoff_hare_when_stag_success
    )
    payoff_stag_fail_values = _parse_float_list(args.payoff_stag_fail)
    payoff_hare_fail_values = _parse_float_list(args.payoff_hare_fail)

    points: list[SweepPoint] = []
    seed_counter = args.seed_start
    for (
        model,
        agents,
        rounds,
        lie_fraction,
        payoff_stag_success,
        payoff_hare_when_stag_success,
        payoff_stag_fail,
        payoff_hare_fail,
    ) in product(
        models,
        num_agents,
        num_rounds,
        lie_fractions,
        payoff_stag_success_values,
        payoff_hare_when_stag_success_values,
        payoff_stag_fail_values,
        payoff_hare_fail_values,
    ):
        if args.stag_thresholds:
            stag_thresholds = _parse_int_list(args.stag_thresholds)
        else:
            stag_thresholds = [agents]
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
                        lie_fraction=lie_fraction,
                        stag_success_threshold=stag_success_threshold,
                        payoff_stag_success=payoff_stag_success,
                        payoff_hare_when_stag_success=payoff_hare_when_stag_success,
                        payoff_stag_fail=payoff_stag_fail,
                        payoff_hare_fail=payoff_hare_fail,
                        order_ablation="a1",
                        adversary_ablation="base",
                        heterogeneity_ablation="h1",
                        h3_liar_policy=args.h3_liar_policy,
                        model_pool_csv="",
                        ablation_code="base",
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

    subset_size = max(1, args.ablation_subset_runs)
    base_subset = points[:subset_size]
    if not base_subset:
        return []

    model_pool_csv = args.model_pool.strip()
    if not model_pool_csv:
        model_pool_csv = args.models

    ablation_points: list[SweepPoint] = list(base_subset)
    for code in ablation_codes:
        for point in base_subset:
            variant = replace(point, ablation_code=code)
            if code == "b3":
                variant = replace(variant, adversary_ablation="b3")
            elif code in {"a1", "a2", "a3"}:
                variant = replace(variant, order_ablation=code)
            elif code == "h1":
                variant = replace(variant, heterogeneity_ablation="h1", model_pool_csv="")
            elif code == "h2":
                variant = replace(
                    variant,
                    heterogeneity_ablation="h2",
                    model_pool_csv=model_pool_csv,
                )
            elif code == "h3":
                variant = replace(
                    variant,
                    heterogeneity_ablation="h3",
                    h3_liar_policy=args.h3_liar_policy,
                    model_pool_csv=model_pool_csv,
                )
            ablation_points.append(variant)

    return ablation_points


def _build_timestamp_name(prefix: str, suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"


def _build_sweep_summary_path(log_dir: Path) -> Path:
    return log_dir / _build_timestamp_name("stag_hunt_sweep", ".csv")


def _build_sweep_points_path(log_dir: Path, configured: str) -> Path:
    if configured:
        return Path(configured)
    return log_dir / _build_timestamp_name("stag_hunt_sweep_points", ".jsonl")


def _save_sweep_points(path: Path, points: list[SweepPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for point in points:
            f.write(json.dumps(asdict(point), sort_keys=True))
            f.write("\n")


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
) -> dict[str, Any]:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    estimates = _estimate_tokens_for_point(
        point=point,
        output_tokens_per_turn=args.output_tokens_per_turn,
        input_overhead_tokens_per_turn=args.input_overhead_tokens_per_turn,
    )
    model_pool = _parse_csv_values(point.model_pool_csv) if point.model_pool_csv else None

    config = GameConfig(
        model=point.model,
        num_agents=point.num_agents,
        num_rounds=point.num_rounds,
        lie_fraction=point.lie_fraction,
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

    logger = InspectLogger(log_dir=str(log_dir), eval_prefix=args.eval_prefix)

    print(
        f"[{run_index}/{run_total}] "
        f"model={point.model} agents={point.num_agents} rounds={point.num_rounds} "
        f"lie_fraction={point.lie_fraction} M={point.stag_success_threshold} "
        f"ablation={point.ablation_code} "
        f"replicate={point.replicate} seed={point.seed} "
        f"est_tokens≈{estimates['estimated_total_tokens']}"
    )

    started = time.perf_counter()
    status = "ok"
    error = ""
    results: dict[str, Any] = {}
    cost_cumulative_before = client.total_cost
    try:
        simulation = StagHuntSimulation(client=client, config=config, logger=logger)
        if args.verbose:
            results = await simulation.run_game()
        else:
            if args.max_concurrency == 1:
                with contextlib.redirect_stdout(io.StringIO()):
                    results = await simulation.run_game()
            else:
                results = await simulation.run_game()

        if getattr(logger, "_log_file", None):
            convert_flashlite_logs_to_inspect(logger._log_file)
    except Exception as exc:  # pragma: no cover - safety path
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        if args.fail_fast:
            raise
    finally:
        logger.close()

    elapsed_sec = time.perf_counter() - started
    csv_exports = results.get("csv_exports", {})
    return {
        "status": status,
        "error": error,
        "model": point.model,
        "num_agents": point.num_agents,
        "num_rounds": point.num_rounds,
        "lie_fraction": point.lie_fraction,
        "stag_success_threshold": point.stag_success_threshold,
        "payoff_stag_success": point.payoff_stag_success,
        "payoff_hare_when_stag_success": point.payoff_hare_when_stag_success,
        "payoff_stag_fail": point.payoff_stag_fail,
        "payoff_hare_fail": point.payoff_hare_fail,
        "order_ablation": point.order_ablation,
        "adversary_ablation": point.adversary_ablation,
        "heterogeneity_ablation": point.heterogeneity_ablation,
        "h3_liar_policy": point.h3_liar_policy,
        "model_pool_csv": point.model_pool_csv,
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

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _build_sweep_summary_path(log_dir)
    points_path = _build_sweep_points_path(log_dir, args.save_sweep_points)
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

    print(f"Sweep runs: {len(points)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Sweep points: {points_path}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(
        "Estimated tokens (all runs): "
        f"input≈{total_est_input:,}, output≈{total_est_output:,}, total≈{total_est_all:,}"
    )
    if args.ablations:
        print(
            "Ablation mode enabled: "
            f"{args.ablations} on first {max(1, args.ablation_subset_runs)} base points"
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

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    total_elapsed = sum(float(row["elapsed_sec"]) for row in rows)
    print(f"Completed sweep: {ok_count}/{len(rows)} successful")
    print(f"Shared client total cost: ${shared_client.total_cost:.4f}")
    print(f"Sum of per-run wall times: {total_elapsed:.1f}s")
    print(f"Wrote sweep summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
