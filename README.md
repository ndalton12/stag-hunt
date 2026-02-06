# Stag Hunt Simulation

This project runs multi-agent Stag Hunt experiments with configurable:

- `N` players and `M` stag-success threshold
- payoff matrix (`stag_success` and `stag_fail` cases)
- adversarial fraction (`lie_fraction`)
- model composition and ablations

It includes:

- `src/stag_hunt/sim.py`: run one simulation
- `src/stag_hunt/sweep_sim.py`: run many simulations over a grid or explicit point set

## Setup

```bash
uv sync
```

## Run a Single Simulation (`sim.py`)

```bash
uv run python -m stag_hunt.sim
```

`sim.py` currently uses defaults defined in `main()` inside `src/stag_hunt/sim.py`.

Important config fields (from `GameConfig`):

- `model`
- `num_agents`
- `num_rounds`
- `lie_fraction`
- `stag_success_threshold` (`M`; if `None`, defaults to `N`)
- `payoff_stag_success`
- `payoff_hare_when_stag_success`
- `payoff_stag_fail`
- `payoff_hare_fail`
- `order_ablation`: `a1`/`a2`/`a3`
- `adversary_ablation`: `base`/`b3`
- `heterogeneity_ablation`: `h1`/`h2`/`h3`
- `h3_liar_policy`: `strongest_liars`/`weakest_liars`
- `model_pool` (used by `h2`/`h3`)
- `seed`

## Run Sweeps (`sweep_sim.py`)

### 1) Grid sweep

```bash
uv run python -m stag_hunt.sweep_sim \
  --models openai/gpt-5-nano \
  --num-agents 4,6 \
  --num-rounds 2 \
  --lie-fractions 0.0,0.25 \
  --stag-thresholds 3,4 \
  --payoff-stag-success 4.0 \
  --payoff-hare-when-stag-success 3.0 \
  --payoff-stag-fail 0.0 \
  --payoff-hare-fail 2.0
```

By default the script prints an estimated token budget and asks for confirmation before execution.

Use `--skip-confirmation` to bypass prompt interaction.

### 2) Explicit sweep points file

You can load points from `.json`, `.jsonl`, or `.csv`:

```bash
uv run python -m stag_hunt.sweep_sim \
  --sweep-points-file ./logs/my_points.jsonl
```

### 3) Save generated sweep points

```bash
uv run python -m stag_hunt.sweep_sim \
  --models openai/gpt-5-nano \
  --num-agents 4 \
  --num-rounds 2 \
  --save-sweep-points ./logs/points.jsonl
```

If `--save-sweep-points` is omitted, a timestamped points file is still written under `logs/`.

## Ablations

Supported ablation codes:

- `b3`: adversarial random-noise action (instead of deterministic flip)
- `a1`: fixed speaking order
- `a2`: random speaking order per round
- `a3`: reverse speaking order
- `h1`: homogeneous model assignment
- `h2`: mixed model assignment from model pool
- `h3`: liars and non-liars assigned opposite ends of model pool

Example:

```bash
uv run python -m stag_hunt.sweep_sim \
  --models openai/gpt-5-mini,openai/gpt-5-nano \
  --num-agents 6 \
  --num-rounds 2 \
  --lie-fractions 0.25 \
  --ablations b3,a2,h2,h3 \
  --ablation-subset-runs 8 \
  --model-pool openai/gpt-5-mini,openai/gpt-5-nano
```

Cost control behavior:

- Ablations run on a smaller base subset (default `--ablation-subset-runs 12`)
- Token estimate is shown before running
- Confirmation prompt is required unless `--skip-confirmation`

## Outputs

All outputs are written under `logs/`.

Per-run files:

- `*_round_metrics.csv`
- `*_agent_metrics.csv`
- `*_agent_text.csv`
- `*_agent_summary.csv`
- inspect-compatible logs

Sweep files:

- `stag_hunt_sweep_*.csv` (sweep summary)
- `stag_hunt_sweep_points_*.jsonl` (effective run list)

Global runs index:

- `logs/stag_hunt_runs.csv`

## Notes

- `sweep_sim.py` uses a shared `Flashlite` client across concurrent runs so RPM/TPM throttling is shared globally.
- `--max-concurrency` controls simultaneous runs.
- For low-cost testing, start with:
  - one model
  - small `num_agents`
  - `num_rounds=1` or `2`
  - low `ablation-subset-runs`
