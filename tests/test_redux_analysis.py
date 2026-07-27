from __future__ import annotations

import unittest

import pandas as pd

from stag_hunt.analysis import SweepData
from stag_hunt.redux_analysis import (
    build_action_channel_table,
    build_enriched_turns,
    configuration_summary,
    hierarchical_summary,
)


class ReduxAnalysisTest(unittest.TestCase):
    def _data(self) -> SweepData:
        run_id = "run_1"
        model = "openai/gpt-5-mini"
        runs = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model": model,
                    "seed": 7,
                    "num_agents": 3,
                    "num_rounds": 1,
                    "num_liars": 1,
                    "stag_success_threshold": 2,
                    "payoff_stag_success": 4.0,
                    "payoff_hare_when_stag_success": 2.0,
                    "payoff_stag_fail": 0.0,
                    "payoff_hare_fail": 2.0,
                    "order_ablation": "a1",
                    "adversary_ablation": "base",
                    "heterogeneity_ablation": "h1",
                }
            ]
        )
        actions = [
            # honest Stag remains public Stag
            (0, "Agent_0", False, True, True, False, 0.0),
            # corrupted Stag becomes public Hare
            (1, "Agent_1", True, True, False, True, 2.0),
            # honest Hare remains public Hare
            (2, "Agent_2", False, False, False, False, 2.0),
        ]
        agent_metrics = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model": model,
                    "round": 1,
                    "turn_index": turn,
                    "agent": agent,
                    "is_liar": is_liar,
                    "original_action": "STAG" if original else "HARE",
                    "original_is_stag": original,
                    "reported_action": "STAG" if public else "HARE",
                    "reported_is_stag": public,
                    "was_flipped": flipped,
                    "stag_success": False,
                    "realized_payoff": payoff,
                    "is_correct": not public,
                    "confidence": 0.8,
                }
                for turn, agent, is_liar, original, public, flipped, payoff in actions
            ]
        )
        return SweepData(
            runs=runs,
            round_metrics=pd.DataFrame(),
            agent_metrics=agent_metrics,
            agent_summary=pd.DataFrame(),
            sweep_points=pd.DataFrame(),
        )

    def test_builds_histories_and_flip_induced_loss(self) -> None:
        turns = build_enriched_turns(self._data())
        last = turns[turns["agent"] == "Agent_2"].iloc[0]
        self.assertEqual(int(last["prior_public_stag_count"]), 1)
        self.assertEqual(int(last["prior_original_stag_count"]), 2)
        self.assertEqual(int(last["prior_liar_count"]), 1)
        self.assertAlmostEqual(float(last["prior_public_stag_share"]), 0.5)
        self.assertAlmostEqual(float(last["prior_original_stag_share"]), 1.0)

        channel = build_action_channel_table(turns)
        self.assertEqual(len(channel), 1)
        row = channel.iloc[0]
        self.assertEqual(int(row["original_stag_count"]), 2)
        self.assertEqual(int(row["public_stag_count"]), 1)
        self.assertEqual(float(row["intended_success"]), 1.0)
        self.assertEqual(float(row["public_success"]), 0.0)
        self.assertEqual(float(row["flip_induced_loss"]), 1.0)
        self.assertAlmostEqual(float(row["honest_original_stag_rate"]), 0.5)
        self.assertAlmostEqual(float(row["honest_truthful_payoff"]), 3.0)
        self.assertAlmostEqual(float(row["honest_realized_payoff"]), 1.0)

    def test_hierarchical_summary_does_not_overweight_repeated_rows(self) -> None:
        frame = pd.DataFrame(
            [
                # Model A, seed 1 appears three times but is one cell.
                *[
                    {
                        "model_short": "a",
                        "num_agents": 5,
                        "stag_success_threshold": 3,
                        "num_liars": 1,
                        "num_rounds": 1,
                        "seed": 1,
                        "liar_share": 0.2,
                        "metric": 1.0,
                    }
                    for _ in range(3)
                ],
                {
                    "model_short": "a",
                    "num_agents": 5,
                    "stag_success_threshold": 3,
                    "num_liars": 1,
                    "num_rounds": 1,
                    "seed": 2,
                    "liar_share": 0.2,
                    "metric": 0.0,
                },
                {
                    "model_short": "b",
                    "num_agents": 5,
                    "stag_success_threshold": 3,
                    "num_liars": 1,
                    "num_rounds": 1,
                    "seed": 1,
                    "liar_share": 0.2,
                    "metric": 1.0,
                },
            ]
        )
        summary = hierarchical_summary(
            frame,
            group_cols=["liar_share"],
            metrics=["metric"],
            bootstrap_replicates=50,
        )
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary.iloc[0]["metric"]), 0.75)
        self.assertEqual(int(summary.iloc[0]["n_models"]), 2)
        self.assertEqual(int(summary.iloc[0]["n_cells"]), 3)

    def test_configuration_summary_retains_model_groups(self) -> None:
        frame = pd.DataFrame(
            [
                *[
                    {
                        "model_short": "a",
                        "num_agents": 5,
                        "stag_success_threshold": 3,
                        "num_liars": 1,
                        "num_rounds": 4,
                        "seed": 1,
                        "liar_share": 0.2,
                        "metric": 1.0,
                    }
                    for _ in range(3)
                ],
                {
                    "model_short": "a",
                    "num_agents": 5,
                    "stag_success_threshold": 3,
                    "num_liars": 1,
                    "num_rounds": 4,
                    "seed": 2,
                    "liar_share": 0.2,
                    "metric": 0.0,
                },
                {
                    "model_short": "b",
                    "num_agents": 5,
                    "stag_success_threshold": 3,
                    "num_liars": 1,
                    "num_rounds": 4,
                    "seed": 1,
                    "liar_share": 0.2,
                    "metric": 0.25,
                },
            ]
        )
        summary = configuration_summary(
            frame,
            group_cols=["model_short", "liar_share"],
            metrics=["metric"],
            bootstrap_replicates=50,
        ).sort_values("model_short")
        self.assertEqual(summary["model_short"].tolist(), ["a", "b"])
        self.assertAlmostEqual(float(summary.iloc[0]["metric"]), 0.5)
        self.assertAlmostEqual(float(summary.iloc[1]["metric"]), 0.25)
        self.assertEqual(summary["n_cells"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
