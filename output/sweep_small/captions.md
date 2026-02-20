# Figure Captions

## Fig 1 — Coordination Success vs. Liar Fraction (`fig1_coordination`)

Stag-success rate as a function of liar fraction, with each line representing a model. Rows correspond to the stag-success threshold *M* (the minimum number of agents that must report STAG for coordination to succeed), and columns correspond to group size *N*. Error bands show 68% CIs. Coordination degrades sharply once liar fraction exceeds roughly 1/*M*, as liars reporting HARE push the group below the coordination threshold.

## Fig 2 — Honest-Agent Accuracy Over Rounds (`fig2_accuracy_trajectory`)

Accuracy of non-liar agents across rounds, with each line representing a liar-fraction level. Faceted by model. Accuracy is computed only over honest agents (liars are excluded since they always report incorrectly by construction). Shows whether honest agents learn or degrade over repeated rounds of potentially corrupted group discussion.

## Fig 3 — Confidence Calibration (`fig3_calibration`)

Scatter of per-agent mean confidence vs. accuracy, split into two panels by role (Honest and Liar). Points are colored by model. The dashed diagonal represents perfect calibration. Small random y-jitter is applied to break up horizontal banding caused by discrete accuracy values (0, 0.25, 0.5, 0.75, 1.0 over 4 rounds). Most agents cluster in the high-confidence region regardless of actual accuracy, indicating systematic overconfidence.

## Fig 4 — Liar Influence (`fig4_influence`)

Mean influence on later-speaking agents, grouped by binned liar fraction and split by role. Influence is defined as the fraction of subsequent agents in the same round whose report matches the focal agent's report. At low liar fractions, honest agents have high influence (most agents agree with the truth). As liar fraction increases, liar influence rises and honest influence falls, showing liars successfully corrupting the information environment.

## Fig 5 — Payoff by Role (`fig5_payoffs`)

Mean realized payoff by liar fraction and role, faceted by model. Error bars show 68% CIs. Honest agents consistently earn higher payoffs than liars, but both roles' payoffs decline as liar fraction increases — reflecting the collective cost of coordination failure. The gap between Honest and Liar payoffs narrows at high liar fractions as coordination collapses for everyone.

## Fig 6 — Accuracy Heatmap (`fig6_heatmap`)

Heatmap of mean overall accuracy across the num_agents × num_liars parameter grid. Rows correspond to stag-success threshold *M*, columns to model. Cell annotations show the mean accuracy value. The green-to-red color scale highlights the transition from high accuracy (few liars) to low accuracy (many liars). Provides a compact overview of the full parameter space.

## Fig 7 — Consensus and Entropy Dynamics (`fig7_consensus_entropy`)

Two-column layout tracking consensus rate (left) and report entropy (right) over rounds, with each line representing a liar-fraction level. Rows correspond to stag-success threshold *M*. Consensus rate is the fraction of agents reporting the majority action. Report entropy (bits) measures diversity in reported actions. At 0% liars, consensus is high and entropy is low. As liar fraction increases, entropy rises and consensus drops, reflecting increased disagreement injected by liars.

## Fig 8 — Persistence by Role (`fig8_persistence`)

Persistence rate (fraction of consecutive rounds where an agent's report stays the same) by liar fraction and role. Rows correspond to stag-success threshold *M*, columns to model. Higher persistence indicates agents are more committed to their position across rounds. Honest agents tend to show higher persistence than liars in low-liar settings, but the pattern can reverse at high liar fractions as honest agents get pulled toward liar-corrupted reports.

## Fig 9 — Coordination Dynamics Over Rounds (`fig9_coordination_dynamics`)

Stag-success rate over rounds, binned by liar fraction, faceted by model. Complements fig 1 (which shows aggregate coordination) by revealing how coordination evolves within a game. Uses liar-fraction quartile bins to reduce visual clutter.

## Fig 10 — Turn-Order Effects / A1 Ablation (`fig10_turn_order`)

Honest-agent accuracy by speaking position, binned by liar fraction, faceted by model.

**Speaking position** refers to the agent's index in the fixed turn order used under the A1 ablation. Under A1, agents always speak in the same sequence each round: position 0 speaks first and sees no prior reports from other agents, position 1 speaks second and sees position 0's report, and so on up to position *N*−1, which sees all prior agents' reports before responding.

**What is measured at each position:** For a given speaking position *X*, the plotted accuracy is the mean of `is_correct` across all honest-agent observations at `turn_index == X`, pooled over all runs and rounds. Which agent occupies position *X* is fixed within a run (agent 0 is always position 0, etc.), but liar assignment varies across runs — so position *X* is sometimes occupied by a liar (excluded from this plot) and sometimes by an honest agent (included). The value at each position thus answers: "when an honest agent happens to speak at position *X*, how accurate are they on average?" Note that positions 0–2 include data from both *N*=3 and *N*=5 runs, while positions 3–4 only include *N*=5 runs.

This figure reveals an information-cascade effect: agents who speak later (higher position) have lower accuracy when liars are present, because they are exposed to more (potentially corrupted) prior reports. At high liar fractions, accuracy drops from ~90% at position 0 to ~30% at position 4. The effect is weaker or absent at low liar fractions, where prior reports are mostly truthful and reinforce rather than undermine accuracy.

## Fig 11 — Confidence Dynamics Over Rounds (`fig11_confidence_dynamics`)

Mean group-level confidence over rounds, by liar fraction, faceted by model. Confidence is the self-reported certainty each agent assigns to its answer (0–1). Despite accuracy degrading under liar influence (fig 2), confidence remains high and stable (0.7–0.9) across all liar fractions and rounds. This suggests agents do not recognize when their information environment has been corrupted — they remain confidently wrong.
