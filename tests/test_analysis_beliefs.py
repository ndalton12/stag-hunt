from __future__ import annotations

import math
import unittest

import pandas as pd

from stag_hunt.analysis import SweepData, build_belief_benchmark


class BuildBeliefBenchmarkTest(unittest.TestCase):
    def test_reconstructs_bayesian_fields_from_public_history(self) -> None:
        run_id = "run_1"
        model = "openai/gpt-5-mini"

        runs = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model": model,
                    "num_agents": 3,
                    "num_liars": 1,
                    "num_rounds": 1,
                    "stag_success_threshold": 2,
                    "payoff_stag_success": 4.0,
                    "payoff_stag_fail": 0.0,
                    "payoff_hare_fail": 2.0,
                    "order_ablation": "a1",
                    "adversary_ablation": "base",
                    "heterogeneity_ablation": "h1",
                }
            ]
        )

        agent_metrics = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model": model,
                    "round": 1,
                    "turn_index": 0,
                    "agent": "Agent_0",
                    "is_liar": False,
                    "original_action": "STAG",
                    "original_is_stag": True,
                    "reported_action": "STAG",
                    "reported_is_stag": True,
                    "is_correct": True,
                    "realized_payoff": 4.0,
                    "num_agents": 3,
                    "num_liars": 1,
                    "liar_share": 1 / 3,
                    "liar_share_bin": "25–50%",
                    "model_short": "gpt-5-mini",
                    "role": "Honest",
                },
                {
                    "run_id": run_id,
                    "model": model,
                    "round": 1,
                    "turn_index": 1,
                    "agent": "Agent_1",
                    "is_liar": False,
                    "original_action": "HARE",
                    "original_is_stag": False,
                    "reported_action": "HARE",
                    "reported_is_stag": False,
                    "is_correct": False,
                    "realized_payoff": 2.0,
                    "num_agents": 3,
                    "num_liars": 1,
                    "liar_share": 1 / 3,
                    "liar_share_bin": "25–50%",
                    "model_short": "gpt-5-mini",
                    "role": "Honest",
                },
                {
                    "run_id": run_id,
                    "model": model,
                    "round": 1,
                    "turn_index": 2,
                    "agent": "Agent_2",
                    "is_liar": True,
                    "original_action": "HARE",
                    "original_is_stag": False,
                    "reported_action": "HARE",
                    "reported_is_stag": False,
                    "is_correct": False,
                    "realized_payoff": 2.0,
                    "num_agents": 3,
                    "num_liars": 1,
                    "liar_share": 1 / 3,
                    "liar_share_bin": "25–50%",
                    "model_short": "gpt-5-mini",
                    "role": "Liar",
                },
            ]
        )

        data = SweepData(
            runs=runs,
            round_metrics=pd.DataFrame(),
            agent_metrics=agent_metrics,
            agent_summary=pd.DataFrame(),
            sweep_points=pd.DataFrame(),
        )

        bench = build_belief_benchmark(data)

        first = bench[bench["agent"] == "Agent_0"].iloc[0]
        self.assertFalse(bool(first["benchmark_defined"]))
        self.assertEqual(int(first["k_stag_seen"]), 0)
        self.assertEqual(int(first["n_observed"]), 0)

        second = bench[bench["agent"] == "Agent_1"].iloc[0]
        expected_q_star = 1 - math.sqrt(2) / 2
        self.assertTrue(bool(second["benchmark_defined"]))
        self.assertAlmostEqual(float(second["q_hat"]), 1.0)
        self.assertAlmostEqual(float(second["q_corrected"]), 1.0)
        self.assertAlmostEqual(float(second["q_star"]), expected_q_star, places=6)
        self.assertEqual(second["rational_action"], "STAG")
        self.assertFalse(bool(second["matches_bayesian"]))
        self.assertTrue(bool(second["false_defect"]))
        self.assertFalse(bool(second["false_cooperate"]))

        honest = bench[bench["role"] == "Honest"]
        eligible = int(honest["benchmark_defined"].sum())
        no_prior = int((~honest["benchmark_defined"]).sum())
        self.assertEqual(len(honest), eligible + no_prior)
        self.assertEqual(eligible, 1)
        self.assertEqual(no_prior, 1)


if __name__ == "__main__":
    unittest.main()
