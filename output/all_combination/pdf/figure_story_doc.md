# Stag Hunt Figure Guide

This document explains what each analysis figure is trying to show, why it exists, and what story to read from it.

## Core coordination and learning figures

### Fig 1 — Coordination Success vs. Liar Fraction
Purpose: Show the main failure mode of the system as adversarial participation increases.

Story: Coordination succeeds reliably at low liar fractions and then degrades as corruption pushes the group below the effective coordination threshold.

### Fig 1 Highlight — N=5, M=3
Purpose: Give one clean focal setting where the coordination transition is easy to inspect.

Story: This is the most presentation-friendly slice of Fig 1 and makes the onset of failure easier to compare across models.

### Fig 2 — Honest-Agent Accuracy & Confidence Over Rounds
Purpose: Compare actual learning quality to self-reported confidence over repeated discussion rounds.

Story: Honest agents can become less accurate even while confidence stays high, which is evidence of confident error under corruption.

### Fig 2 Alternate — Combined Models
Purpose: Put all models on the same figure so cross-model differences are direct.

Story: This emphasizes relative model robustness rather than within-model liar-fraction effects.

### Fig 2 Alternate B — Focused Confidence Gap
Purpose: Highlight the gap between honest confidence and honest accuracy.

Story: This is a compact view of overconfidence dynamics. Positive values mean agents are more confident than their actual accuracy warrants.

### Fig 3 — Expected Calibration Error
Purpose: Summarize whether honest-agent confidence tracks actual correctness.

Story: Lower ECE means confidence is informative; higher ECE means agents are systematically miscalibrated.

## Strategic influence and outcome figures

### Fig 4 — Liar Influence
Purpose: Measure how much earlier speakers shape later speakers' reports.

Story: As liar pressure increases, influence shifts away from honest agents and toward deceptive or corrupted reports.

### Fig 4 Alternate — Liar-Honest Influence Gap
Purpose: Collapse Fig 4 into one signed comparison.

Story: Positive values mean liars are more influential than honest agents in that setting.

### Fig 5 — Honest-Agent Payoffs
Purpose: Connect coordination quality to realized utility for honest players.

Story: As corruption rises, honest agents lose payoff even when they are not the ones lying.

### Fig 6 — Accuracy Heatmap
Purpose: Show the broad shape of performance over the parameter grid.

Story: This is the fastest overview of where the system is robust versus brittle.

### Fig 7 — Consensus Dynamics
Purpose: Track how quickly honest-agent consensus forms or collapses over rounds.

Story: Stable consensus at low liar pressure can give way to fragmentation or corrupted consensus at higher liar pressure.

### Fig 9 — Coordination Dynamics Over Rounds
Purpose: Show whether coordination improves, stalls, or collapses within a game.

Story: This complements Fig 1 by adding temporal structure rather than only aggregate success rates.

### Fig 10 — Turn-Order Effects
Purpose: Identify whether speaking later helps or hurts honest agents.

Story: Later speakers can benefit from more information when reports are trustworthy, but they can also become more exposed to corrupted cascades.

## Ablation tables

### Fig 11 — B3 Matched Table
Purpose: Compare random-noise adversaries against the base adversary under matched settings.

Story: This isolates how much of the effect is specific to deterministic flipping rather than adversarial corruption in general.

### Fig 12 — H1 vs H2 Matched Table
Purpose: Compare homogeneous groups with mixed-model groups under matched parameter points.

Story: This tests whether heterogeneity changes coordination success after controlling for the rest of the setup.

### Fig 13 — H1 vs H3 Matched Table
Purpose: Compare homogeneous groups with asymmetric liar/non-liar model assignment.

Story: This tests whether giving liars a model-strength advantage or disadvantage changes outcomes.

## Public-belief benchmark figures

### Fig 14 — Honest-Agent Action vs Belief-Rule Margin
Purpose: Compare the honest-agent response curves implied by the three benchmark rules on a common set of axes.

Story: The key comparison is whether the carryover and trust-weighted rules produce cleaner threshold-like response curves than the naive aggregate rule.

How Fig 14 changed:
- The first version showed only the naive aggregate rule.
- The current version overlays the naive aggregate, carryover, and trust-weighted response curves inside each model panel.
- This makes the figure a direct visual comparison rather than a single-rule diagnostic.

### Fig 15 — Naive-Aggregate Agreement by Turn
Purpose: Show where in the speaking order honest agents align with the naive aggregate rule.

Story: This identifies whether deviations from the naive aggregate rule are concentrated among early speakers, late speakers, or uniformly across the round.

### Fig 16 — Naive-Aggregate Alignment Table
Purpose: Give a compact model-by-model summary of naive-aggregate agreement and its consequences.

Story: This table shows who matches the naive aggregate rule most often, what kinds of mistakes they make, and whether mismatches are costly in correctness or payoff terms.

### Fig 17 — Update-Rule Comparison Table
Purpose: Compare the naive aggregate rule against the two richer alternatives on a common-support match-rate metric in text-table form.

Story: This is the direct model-selection figure for the benchmark family. Higher values mean the rule better matches honest-agent observed behavior on turns where all three rules are comparable.

## Candidate Alternative Belief Rules

These are candidate extensions to the naive aggregate benchmark. The two strongest variants now appear as overlaid curves in Fig 14, with a direct comparison table in Fig 17.

Notation:
- Let \(y_j^t \in \{0,1\}\) denote agent \(j\)'s public report in round \(t\), where \(1=\mathrm{STAG}\).
- Let \(S_i^t\) be the set of speakers agent \(i\) has observed so far in round \(t\).
- Let \(q_i^t\) denote agent \(i\)'s belief about the public STAG report rate.
- Let \(\hat q_i^t\) denote the rule-implied estimate of that public belief used in the benchmark figures below.
- In this section, beliefs are about the transcript-level / public report process, not a latent true-action rate.

### 1. Carryover-Prior Update

Start each round with a prior anchored in the previous round's posterior:

\[
q_{i,\mathrm{prior}}^{t} = \lambda \hat q_{i,\mathrm{post}}^{t-1} + (1-\lambda) q_0
\]

where \(q_0 \in [0,1]\) is a neutral baseline prior over public STAG reports and \(\lambda \in [0,1]\) controls memory strength.

Update within the round using pseudo-count pooling:

\[
\hat q_i^t = \frac{\tau q_{i,\mathrm{prior}}^{t} + \sum_{j \in S_i^t} y_j^t}{\tau + |S_i^t|}
\]

where \(\tau > 0\) is the prior strength.

Story: agents assume the public reporting environment persists across rounds, but revise that prior as current-round reports arrive.

### 2. Agent-Specific Trust Update

Instead of weighting all speakers equally, let agent \(i\) maintain a trust weight \(\rho_{ij}^t \in [0,1]\) for each other agent \(j\). Here \(\rho_{ij}^t\) should be interpreted as a predictive weight on how informative \(j\)'s public reports are about future public STAG reporting, not as an attempt to recover a hidden true action. Then form a weighted within-round estimate:

\[
\hat q_i^t = \frac{\tau q_{i,\mathrm{prior}}^{t} + \sum_{j \in S_i^t} \rho_{ij}^t y_j^t}{\tau + \sum_{j \in S_i^t} \rho_{ij}^t}
\]

Update trust across rounds using a simple Beta-style reliability score. Let \(z_j^t \in \{0,1\}\) indicate whether agent \(j\)'s public report in round \(t\) matched some target notion of reliability, such as the final round outcome or the final majority report. Maintain

\[
a_{ij}^{t+1} = a_{ij}^{t} + z_j^t, \qquad b_{ij}^{t+1} = b_{ij}^{t} + (1-z_j^t)
\]

and define the trust weight as

\[
\rho_{ij}^{t+1} = \frac{a_{ij}^{t+1}}{a_{ij}^{t+1} + b_{ij}^{t+1}}
\]

Story: agents learn not just how STAG-heavy the public report environment is, but which specific speakers are worth weighting when forecasting future public reports.

### Recommendation for Discussion

If the goal is to present a small, interpretable family of richer benchmarks, the strongest comparison set is:

1. Naive aggregate public-belief rule
2. Carryover-prior update
3. Agent-specific trust update

These span three distinct ideas:
- no memory beyond the current round
- generic memory of the public reporting environment
- memory about individual speaker reliability rather than only aggregate group state
