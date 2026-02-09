"""
Stag Hunt Game with Information Corruption
==========================================

Demonstrates a multi-agent LLM simulation studying equilibrium stability
in iterated Stag Hunt games under controlled information corruption.

Built on flashlite's MultiAgentChat:
- Each agent has a system prompt defining the game rules
- Round signals are delivered publicly by GameMaster
- Structured outputs (AgentDecision) are enforced via response_model
- "Lying" is implemented by programmatically flipping reported_action
  and patching the public transcript

To run:
    uv run python -m stag_hunt.sim
"""

import asyncio
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from flashlite import (
    Agent,
    CompletionRequest,
    CompletionResponse,
    Flashlite,
    InspectLogger,
    MultiAgentChat,
    RateLimitConfig,
    UsageInfo,
    parse_json_response,
)
from flashlite.observability import convert_flashlite_logs_to_inspect
from pydantic import BaseModel, Field

from stag_hunt.results import (
    SharedCSVPaths,
    append_csv_results,
    build_agent_summary_rows,
    build_round_metric_rows,
    export_csv_results,
    make_run_context,
)

# Prompts directory for Flashlite templates (used by the lying mechanism)
PROMPTS_DIR = Path(__file__).parent / "prompts"

# ============================================================================
# Structured Output Model
# ============================================================================


class AgentDecision(BaseModel):
    """Structured output for each agent's decision."""

    reported_action: Literal["STAG", "HARE"] = Field(
        description="Your public action/report for this round: STAG or HARE"
    )
    confidence: float = Field(
        ge=0, le=1, description="Your confidence in this assessment (0-1)"
    )
    justification: str = Field(description="One sentence explaining your reasoning")


# ============================================================================
# Game Configuration
# ============================================================================


@dataclass
class GameConfig:
    """Configuration for the Stag Hunt simulation."""

    model: str
    num_agents: int = 5
    num_rounds: int = 3
    num_liars: int = 1  # Number of agents that will have signals flipped
    stag_success_threshold: int | None = None  # M in n-player stag hunt
    payoff_stag_success: float = 4.0
    payoff_hare_when_stag_success: float = 3.0
    payoff_stag_fail: float = 0.0
    payoff_hare_fail: float = 2.0
    order_ablation: Literal["a1", "a2", "a3"] = "a1"
    adversary_ablation: Literal["base", "b3"] = "base"
    heterogeneity_ablation: Literal["h1", "h2", "h3"] = "h1"
    h3_liar_policy: Literal["strongest_liars", "weakest_liars"] = "strongest_liars"
    model_pool: list[str] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate configuration and payoff structure."""
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if (
            self.stag_success_threshold is not None
            and self.stag_success_threshold > self.num_agents
        ):
            raise ValueError("stag_success_threshold must be <= num_agents")
        if self.stag_success_threshold is not None and self.stag_success_threshold < 1:
            raise ValueError("stag_success_threshold must be >= 1")
        if not 0 <= self.num_liars <= self.num_agents:
            raise ValueError("num_liars must be between 0 and num_agents")

        # Enforce standard Stag Hunt ordering:
        # u(Stag,Stag) > u(Hare,Hare) > u(Hare,Stag) > u(Stag,Hare)
        # Mapped to this N-player parameterization as:
        # payoff_stag_success > payoff_hare_fail >
        # payoff_hare_when_stag_success > payoff_stag_fail
        if not (
            self.payoff_stag_success
            > self.payoff_hare_fail
            >= self.payoff_hare_when_stag_success
            > self.payoff_stag_fail
        ):
            raise ValueError(
                "Invalid payoffs: require "
                "payoff_stag_success > payoff_hare_fail > "
                "payoff_hare_when_stag_success > payoff_stag_fail"
            )


# ============================================================================
# Constants
# ============================================================================

AGENT_SYSTEM_PROMPT = """\
You are {name}, an agent in a Stag Hunt coordination game.

In each round, you receive a public signal from GameMaster suggesting which \
equilibrium (STAG or HARE) is payoff-dominant. You must decide what ACTION to \
report publicly.

This is an N-player game with N={num_agents} and threshold M={stag_success_threshold}.
If at least M players choose STAG, the stag hunt succeeds.

Payoffs:
- If stag succeeds (>= M STAG):
  - STAG chooser gets {payoff_stag_success}
  - HARE chooser gets {payoff_hare_when_stag_success}
- If stag fails (< M STAG):
  - STAG chooser gets {payoff_stag_fail}
  - HARE chooser gets {payoff_hare_fail}

Your goal is to help the group coordinate on the best equilibrium. \
Consider what others have reported and whether you trust them."""


# ============================================================================
# Simulation
# ============================================================================


@dataclass
class AgentRole:
    """Tracks an agent's role in the simulation (separate from the chat Agent)."""

    name: str
    model: str
    is_liar: bool = False


class StagHuntSimulation:
    """
    Manages a Stag Hunt game simulation using MultiAgentChat.

    Key architecture:
    - A ``MultiAgentChat`` orchestrates the conversation between agents.
    - Each agent's system prompt defines the game rules and their identity.
    - A "GameMaster" injects round announcements and public round signals.
    - Structured outputs (``AgentDecision``) are extracted via ``speak()``
      with ``response_model``.
    - Lying is applied post-hoc: the signal is flipped, a new justification
      is generated via the ``justify_lie`` template, and the public
      transcript entry is patched so subsequent agents see the flipped version.
    """

    def __init__(
        self,
        client: Flashlite,
        config: GameConfig,
        logger: InspectLogger | None = None,
        run_id_override: str | None = None,
        shared_csv_paths: SharedCSVPaths | None = None,
    ):
        self.client = client
        self.config = config
        random.seed(self.config.seed)
        self.logger = logger
        self.shared_csv_paths = shared_csv_paths
        self.chat = MultiAgentChat(client, default_model=config.model)
        self.roles: list[AgentRole] = []
        self.round_data: list[dict] = []
        self.run = make_run_context(
            output_dir=Path("./logs"),
            logger=self.logger,
            model=self.config.model,
            seed=self.config.seed,
            config_dict=asdict(self.config),
            run_id_override=run_id_override,
        )
        self._setup_agents()

    # -- Setup -------------------------------------------------------------

    def _setup_agents(self) -> None:
        """Create chat agents and randomly assign liar roles."""
        num_liars = self.config.num_liars
        liar_indices = set(random.sample(range(self.config.num_agents), num_liars))
        stag_success_threshold = self._resolve_stag_success_threshold()
        model_pool = self._resolve_model_pool()

        for i in range(self.config.num_agents):
            name = f"Agent_{i}"
            is_liar = i in liar_indices
            agent_model = self._pick_agent_model(
                agent_index=i,
                is_liar=is_liar,
                model_pool=model_pool,
            )
            self.roles.append(AgentRole(name=name, model=agent_model, is_liar=is_liar))
            self.chat.add_agent(
                Agent(
                    name=name,
                    model=agent_model,
                    system_prompt=AGENT_SYSTEM_PROMPT.format(
                        name=name,
                        num_agents=self.config.num_agents,
                        stag_success_threshold=stag_success_threshold,
                        payoff_stag_success=self.config.payoff_stag_success,
                        payoff_hare_when_stag_success=(
                            self.config.payoff_hare_when_stag_success
                        ),
                        payoff_stag_fail=self.config.payoff_stag_fail,
                        payoff_hare_fail=self.config.payoff_hare_fail,
                    ),
                )
            )

    def _resolve_model_pool(self) -> list[str]:
        """Resolve and validate model pool used for heterogeneity ablations."""
        if self.config.model_pool:
            pool = [m.strip() for m in self.config.model_pool if m.strip()]
            if pool:
                return pool
        return [self.config.model]

    def _pick_agent_model(
        self, agent_index: int, is_liar: bool, model_pool: list[str]
    ) -> str:
        """Choose model per agent according to heterogeneity ablation."""
        het = self.config.heterogeneity_ablation
        if het == "h1" or len(model_pool) == 1:
            return self.config.model

        if het == "h2":
            return model_pool[agent_index % len(model_pool)]

        strongest = model_pool[0]
        weakest = model_pool[-1]
        if self.config.h3_liar_policy == "strongest_liars":
            return strongest if is_liar else weakest
        return weakest if is_liar else strongest

    def _resolve_stag_success_threshold(self) -> int:
        """Get threshold M (minimum STAG picks needed for stag success)."""
        if self.config.stag_success_threshold is None:
            return self.config.num_agents
        threshold = self.config.stag_success_threshold
        if threshold < 1 or threshold > self.config.num_agents:
            raise ValueError(
                "stag_success_threshold must be between 1 and num_agents (inclusive)"
            )
        return threshold

    def _get_role(self, name: str) -> AgentRole:
        """Look up a role by agent name."""
        return next(r for r in self.roles if r.name == name)

    @property
    def liar_names(self) -> list[str]:
        """Names of agents designated as liars."""
        return [r.name for r in self.roles if r.is_liar]

    # -- Lying mechanism ---------------------------------------------------

    async def _apply_lying(
        self, role: AgentRole, decision: AgentDecision
    ) -> AgentDecision:
        """Apply adversarial behavior and regenerate justification when action changes."""
        if not role.is_liar:
            return decision

        if self.config.adversary_ablation == "b3":
            target_action: Literal["STAG", "HARE"] = random.choice(["STAG", "HARE"])
        else:
            target_action = "HARE" if decision.reported_action == "STAG" else "STAG"

        if target_action == decision.reported_action:
            return decision

        # Use the justify_lie template to generate a plausible justification
        response = await self.client.complete(
            model=role.model,
            template="justify_lie",
            variables={
                "agent_name": role.name,
                "original_action": decision.reported_action,
                "original_justification": decision.justification,
                "flipped_action": target_action,
            },
            temperature=1.0,
            response_format={"type": "json_object"},
        )

        parsed = parse_json_response(response.content)
        new_justification = parsed.get("justification", decision.justification)

        return AgentDecision(
            reported_action=target_action,
            confidence=decision.confidence,
            justification=new_justification,
        )

    def _ordered_roles_for_round(self) -> list[AgentRole]:
        """Determine speaking order for the current round."""
        if self.config.order_ablation == "a1":
            return list(self.roles)
        if self.config.order_ablation == "a2":
            shuffled = list(self.roles)
            random.shuffle(shuffled)
            return shuffled
        return list(reversed(self.roles))

    def _format_public_message(self, role: AgentRole, decision: AgentDecision) -> str:
        """Format transcript content so speaker identity is explicit."""
        return f"{role.name}: {decision.model_dump_json()}"

    def _patch_transcript(
        self, role: AgentRole, public_decision: AgentDecision
    ) -> None:
        """Replace the last transcript entry with the public, name-prefixed output."""
        self.chat._transcript[-1].content = self._format_public_message(
            role, public_decision
        )

    # -- Inspect logging ---------------------------------------------------

    def _log_agent_turn(
        self,
        role: AgentRole,
        round_num: int,
        sample_id: int,
        input_messages: list[dict],
        decision: AgentDecision,
    ) -> None:
        """Log a single agent turn so Inspect shows each agent's perspective.

        ``input_messages`` should be captured *before* ``speak()`` so they
        reflect exactly what the agent saw when making its decision.
        ``decision`` is the structured output returned directly by ``speak()``.
        """
        if self.logger is None:
            return

        # Metadata (model, token counts) lives on the transcript entry
        last_msg = self.chat.transcript[-1]
        meta = last_msg.metadata

        request = CompletionRequest(
            model=meta.get("model", self.config.model),
            messages=input_messages,
            temperature=1.0,
        )

        response = CompletionResponse(
            content=decision.model_dump_json(),
            model=meta.get("model", self.config.model),
            usage=UsageInfo(
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                total_tokens=meta.get("tokens", 0),
            ),
        )

        self.logger.log(
            request=request,
            response=response,
            sample_id=sample_id,
            epoch=round_num - 1,  # 0-indexed
            metadata={
                "agent": role.name,
                "is_liar": role.is_liar,
                "agent_model": role.model,
            },
        )

    # -- Signal injection --------------------------------------------------

    def _inject_round_signals(
        self, round_num: int, true_action: Literal["STAG", "HARE"]
    ) -> None:
        """Announce the round and broadcast one public signal for all agents."""
        threshold_m = self._resolve_stag_success_threshold()

        # Public announcement everyone sees
        self.chat.add_message(
            "GameMaster",
            f"Round {round_num} begins. Make your decision.",
        )

        # Shared signal everyone sees
        self.chat.add_message(
            "GameMaster",
            (
                f"Public signal for this round: {true_action} is payoff-dominant. "
                "Based on this signal and the public reports from others, "
                "what action do you report? "
                f"Remember: N={self.config.num_agents}, M={threshold_m}."
            ),
        )

    # -- Round execution ---------------------------------------------------

    async def run_round(
        self, round_num: int, true_action: Literal["STAG", "HARE"]
    ) -> list[dict]:
        """Run a single round: inject signals, collect decisions, apply lying."""
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num} — True action signal: {true_action}")
        print("=" * 60)

        self._inject_round_signals(round_num, true_action)

        round_results: list[dict] = []

        speaking_order = self._ordered_roles_for_round()
        for idx, role in enumerate(speaking_order):
            # Snapshot the agent's message view *before* they speak
            input_messages = self.chat.get_messages_for(role.name)

            # Agent speaks — structured output automatically validated
            decision: AgentDecision = await self.chat.speak(
                role.name,
                response_model=AgentDecision,
                temperature=1.0,
            )

            # Log the turn (captures each agent's unique perspective)
            sample_id = (round_num - 1) * len(self.roles) + idx

            # Apply programmatic lying (flips action + regenerates justification)
            public_decision = await self._apply_lying(role, decision)
            was_flipped = decision.reported_action != public_decision.reported_action

            # Patch the public transcript so subsequent agents always see
            # an explicit speaker prefix and any applied corruption.
            self._patch_transcript(role, public_decision)

            self._log_agent_turn(
                role, round_num, sample_id, input_messages, public_decision
            )

            result = {
                "run_id": self.run.run_id,
                "agent": role.name,
                "agent_model": role.model,
                "round": round_num,
                "turn_index": idx,
                "is_liar": role.is_liar,
                "true_action": true_action,
                "true_is_stag": true_action == "STAG",
                "original_action": decision.reported_action,
                "original_is_stag": decision.reported_action == "STAG",
                "reported_action": public_decision.reported_action,
                "reported_is_stag": public_decision.reported_action == "STAG",
                "was_flipped": was_flipped,
                "is_correct": public_decision.reported_action == true_action,
                "confidence": public_decision.confidence,
                "justification": public_decision.justification,
            }
            round_results.append(result)

            # Print result
            liar_tag = " [LIAR]" if role.is_liar else ""
            flip_tag = " (FLIPPED)" if was_flipped else ""
            print(
                f"  {role.name}{liar_tag}: {public_decision.reported_action}{flip_tag} "
                f"(conf: {public_decision.confidence:.2f}) — "
                f'"{public_decision.justification}"'
            )

        threshold_m = self._resolve_stag_success_threshold()
        num_stag = sum(1 for r in round_results if r["reported_is_stag"])
        stag_success = num_stag >= threshold_m
        for result in round_results:
            if stag_success:
                payoff = (
                    self.config.payoff_stag_success
                    if result["reported_is_stag"]
                    else self.config.payoff_hare_when_stag_success
                )
            else:
                payoff = (
                    self.config.payoff_stag_fail
                    if result["reported_is_stag"]
                    else self.config.payoff_hare_fail
                )
            result["realized_payoff"] = payoff
            result["stag_success"] = stag_success
            result["stag_success_threshold"] = threshold_m
            result["num_stag_reported"] = num_stag

        self.round_data.append(
            {
                "round": round_num,
                "true_action": true_action,
                "true_is_stag": true_action == "STAG",
                "stag_success_threshold": threshold_m,
                "num_stag_reported": num_stag,
                "stag_success": stag_success,
                "results": round_results,
            }
        )
        return round_results

    # -- Full game ---------------------------------------------------------

    async def run_game(self) -> dict:
        """Run the full multi-round simulation."""
        print("\n" + "=" * 60)
        print("STAG HUNT SIMULATION")
        print("=" * 60)
        print(f"Agents: {self.config.num_agents}")
        print(f"Liars:  {len(self.liar_names)}")
        print(f"Order ablation: {self.config.order_ablation}")
        print(f"Adversary ablation: {self.config.adversary_ablation}")
        print(f"Heterogeneity ablation: {self.config.heterogeneity_ablation}")
        if self.config.heterogeneity_ablation == "h3":
            print(f"H3 liar policy: {self.config.h3_liar_policy}")
        if self.config.model_pool:
            print(f"Model pool: {', '.join(self.config.model_pool)}")
        print(f"Stag success threshold (M): {self._resolve_stag_success_threshold()}")
        print(
            "Payoffs: "
            f"stag_success[STAG={self.config.payoff_stag_success}, "
            f"HARE={self.config.payoff_hare_when_stag_success}], "
            f"stag_fail[STAG={self.config.payoff_stag_fail}, "
            f"HARE={self.config.payoff_hare_fail}]"
        )
        print(f"Seed: {self.config.seed}")
        print(f"Run ID: {self.run.run_id}")
        print(f"Git commit: {self.run.git_commit}")
        if self.liar_names:
            print(f"Designated liars: {', '.join(self.liar_names)}")

        for round_num in range(1, self.config.num_rounds + 1):
            true_action: Literal["STAG", "HARE"] = (
                "STAG" if round_num % 2 == 1 else "HARE"
            )
            await self.run_round(round_num, true_action)

        return self._analyze_results()

    # -- Analysis ----------------------------------------------------------

    def _analyze_results(self) -> dict:
        """Analyze the simulation results and print a summary."""
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        total_correct = 0
        total_decisions = 0
        liar_correct = 0
        liar_total = 0

        for rd in self.round_data:
            for result in rd["results"]:
                total_decisions += 1
                if result["is_correct"]:
                    total_correct += 1
                if result["is_liar"]:
                    liar_total += 1
                    if result["is_correct"]:
                        liar_correct += 1

        accuracy = total_correct / total_decisions if total_decisions else 0
        liar_accuracy = liar_correct / liar_total if liar_total else 0

        print(f"Overall accuracy (public reports match truth): {accuracy:.1%}")
        print(f"Liar accuracy (after flipping): {liar_accuracy:.1%}")
        print(f"Total decisions: {total_decisions}")
        if total_decisions:
            avg_payoff = (
                sum(
                    r["realized_payoff"]
                    for rd in self.round_data
                    for r in rd["results"]
                )
                / total_decisions
            )
            print(f"Average realized payoff per decision: {avg_payoff:.3f}")

        # Print conversation stats from MultiAgentChat
        stats = self.chat.stats
        print(f"Total tokens used: {stats['total_tokens']}")
        round_metrics = build_round_metric_rows(self.round_data, self.run)
        agent_summary = build_agent_summary_rows(self.round_data, self.roles, self.run)

        # Use append mode for shared CSV files (sweep), or write mode for standalone
        export_kwargs = {
            "round_data": self.round_data,
            "run": self.run,
            "round_metrics_rows": round_metrics,
            "agent_summary_rows": agent_summary,
            "num_agents": self.config.num_agents,
            "num_rounds": self.config.num_rounds,
            "num_liars": self.config.num_liars,
            "order_ablation": self.config.order_ablation,
            "adversary_ablation": self.config.adversary_ablation,
            "heterogeneity_ablation": self.config.heterogeneity_ablation,
            "h3_liar_policy": self.config.h3_liar_policy,
            "model_pool": "|".join(self._resolve_model_pool()),
            "stag_success_threshold": self._resolve_stag_success_threshold(),
            "payoff_stag_success": self.config.payoff_stag_success,
            "payoff_hare_when_stag_success": self.config.payoff_hare_when_stag_success,
            "payoff_stag_fail": self.config.payoff_stag_fail,
            "payoff_hare_fail": self.config.payoff_hare_fail,
            "accuracy": accuracy,
            "liar_accuracy": liar_accuracy,
        }

        if self.shared_csv_paths is not None:
            csv_exports = append_csv_results(
                shared_paths=self.shared_csv_paths,
                **export_kwargs,
            )
        else:
            csv_exports = export_csv_results(**export_kwargs)

        print(f"Round metrics CSV: {csv_exports['round_metrics']}")
        print(f"Agent metrics CSV: {csv_exports['agent_metrics']}")
        print(f"Agent text CSV: {csv_exports['agent_text']}")
        print(f"Agent summary CSV: {csv_exports['agent_summary']}")
        print(f"Runs index CSV: {csv_exports['runs_index']}")

        return {
            "rounds": self.round_data,
            "accuracy": accuracy,
            "liar_accuracy": liar_accuracy,
            "total_decisions": total_decisions,
            "chat_stats": stats,
            "round_metrics": round_metrics,
            "agent_summary": agent_summary,
            "csv_exports": csv_exports,
        }


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run the Stag Hunt simulation."""
    client = Flashlite(
        default_model="openai/gpt-5-nano",
        template_dir=PROMPTS_DIR,
        track_costs=True,
        rate_limit=RateLimitConfig(requests_per_minute=30, tokens_per_minute=20000),
    )

    # Inspect-compatible logging — each agent's perspective is recorded
    logger = InspectLogger(
        log_dir="./logs",
        eval_prefix="stag_hunt_simulation",
    )

    config = GameConfig(
        num_agents=4,
        num_rounds=2,
        num_liars=1,
        stag_success_threshold=3,
        payoff_stag_success=4.0,
        payoff_hare_when_stag_success=3.0,
        payoff_stag_fail=0.0,
        payoff_hare_fail=2.0,
        order_ablation="a1",
        adversary_ablation="base",
        heterogeneity_ablation="h1",
        model="openai/gpt-5-nano",
    )

    try:
        simulation = StagHuntSimulation(client, config, logger=logger)
        results = await simulation.run_game()

        print(f"\nTotal cost: ${client.total_cost:.4f}")
        print(f"Inspect logs: {logger.log_file}")
        print("\n--- Transcript ---")
        print(simulation.chat.format_transcript(include_metadata=True))

        convert_flashlite_logs_to_inspect(logger._log_file)

        return results
    finally:
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())
