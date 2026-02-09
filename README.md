# Stag Hunt Simulation

This project runs multi-agent Stag Hunt experiments with configurable:

- `N` players, `num_liars`, and `M` stag-success threshold
- payoff matrix (`stag_success` and `stag_fail` cases)
- adversarial count (`num_liars`)
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
- `num_liars`
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

Payoff ordering constraint (enforced):

- `payoff_stag_success > payoff_hare_fail > payoff_hare_when_stag_success > payoff_stag_fail`

This corresponds to the standard Stag Hunt ordering:

- `u(Stag,Stag) > u(Hare,Hare) > u(Hare,Stag) > u(Stag,Hare)`

## Model vs Model Pool

`model` and `model_pool` serve different purposes:

- `model`: base/default model for the run
- `model_pool`: optional list used only for heterogeneity modes (`h2`, `h3`)

How model assignment is computed per agent:

- `h1` (homogeneous):
  - every agent uses `model`
- `h2` (mixed):
  - agents are assigned round-robin over `model_pool` by agent index
  - `agent_model = model_pool[agent_index % len(model_pool)]`
- `h3` (adversarial asymmetry):
  - pool order defines strength ends:
    - `strongest = model_pool[0]`
    - `weakest = model_pool[-1]`
  - `h3_liar_policy=strongest_liars`:
    - liars use `strongest`, non-liars use `weakest`
  - `h3_liar_policy=weakest_liars`:
    - liars use `weakest`, non-liars use `strongest`

Fallback behavior:

- if `model_pool` is missing/empty, code falls back to `[model]`
- with a single-model pool, `h2`/`h3` effectively become homogeneous

Sweep-specific note:

- in `sweep_sim.py`, `--model-pool` is mainly used when expanding `h2/h3` ablations
- if `--model-pool` is not provided, ablation expansion falls back to `--models`
- in saved sweep-point CSVs, `model_pool` is a comma-separated list for that run

## Run Sweeps (`sweep_sim.py`)

### 1) Grid sweep

```bash
uv run python -m stag_hunt.sweep_sim \
  --models openai/gpt-5-nano \
  --agent-configs "4,0;4,1;6,0;6,1;6,2" \
  --num-rounds 2 \
  --stag-thresholds 3,4 \
  --payoffs "4.0,3.0,0.0,2.0"
```

`--agent-configs` accepts semicolon-separated `num_agents,num_liars` pairs.
`--payoffs` accepts semicolon-separated `stag_success,hare_when_stag_success,stag_fail,hare_fail` tuples.

By default the script prints an estimated token budget and asks for confirmation before execution.

Use `--skip-confirmation` to bypass prompt interaction.

### 2) Explicit sweep points file

You can load points from `.csv`:

```bash
uv run python -m stag_hunt.sweep_sim \
  --sweep-points-file ./logs/my_points.csv
```

### 3) Save generated sweep points

```bash
uv run python -m stag_hunt.sweep_sim \
  --models openai/gpt-5-nano \
  --agent-configs "4,1" \
  --num-rounds 2 \
  --save-sweep-points ./logs/points.csv
```

If `--save-sweep-points` is omitted, a timestamped CSV points file is written under `logs/`.

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
  --agent-configs "6,1;6,2" \
  --num-rounds 2 \
  --ablations b3,a2,h2,h3 \
  --ablation-subset-runs 8 \
  --model-pool openai/gpt-5-mini,openai/gpt-5-nano
```

Cost control behavior:

- Ablations run on a smaller random subset of base parameter points (default `--ablation-subset-runs 12`)
  - subset sampling is reproducible from `--seed-start`
  - when `--replicates > 1`, all replicates for each sampled parameter point are included
- Token estimate is shown before running
- Confirmation prompt is required unless `--skip-confirmation`
- Sweep points are pre-validated before execution; invalid points are warned and skipped

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
- `stag_hunt_sweep_points_*.csv` (effective run list)

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
