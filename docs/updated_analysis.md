# Updated Analysis Plan

## Purpose

This plan addresses the central reviewer concern: the current paper measures coordination using public, post-flip actions, so the reported decline in success may combine two different mechanisms:

1. an internal-choice channel, in which agents respond to corrupted transcripts by intending to play Hare; and
2. a mechanical channel, in which an intended action is programmatically flipped before it enters the transcript and outcome calculation.

No new LLM simulations are required. The existing agent-level logs in `logs/all_combination` contain both `original_action` and `reported_action`, as well as role, round, speaking position, confidence, payoff, and outcome. These fields are sufficient for the analyses below.

The proposed revision should replace the broad claim that corruption produces an emergent belief-driven tipping point with a more precise claim:

> Corrupted communication affects coordination through both endogenous changes in agents' intended choices and a direct execution/reporting effect. The direct effect dominates the sharp system-level collapse at high corruption, while honest agents also adapt their choices to the corrupted public history.

## Core notation

For agent $i$, run $r$, and round $t$, define:

- $a_{irt} \in \{0,1\}$: original, pre-flip action, where 1 denotes Stag;
- $\tilde a_{irt} \in \{0,1\}$: public, post-flip action;
- $M_r$: the Stag threshold for the run;
- $F_r/N_r$: liar fraction.

At the run-round level, define:

\[
S^{\mathrm{int}}_{rt}
=
\mathbb{1}\left[\sum_i a_{irt} \ge M_r\right]
\]

and

\[
S^{\mathrm{pub}}_{rt}
=
\mathbb{1}\left[\sum_i \tilde a_{irt} \ge M_r\right].
\]

The first is the outcome that would be obtained if the recorded original choices were implemented truthfully, holding those choices fixed. The second is the outcome actually used by the simulation.

`S_int` is not a no-corruption counterfactual: later original choices may already have been affected by earlier corrupted messages. It isolates the final mechanical flip from the decisions made under the observed transcript.

## Analysis 1: Intended choices, public actions, and outcomes

### Objective

Directly answer the reviewers by separately reporting internal choices, public choices, and threshold outcomes.

### Derived run-round variables

For every base-condition run-round, calculate:

- number and fraction of original Stag choices;
- number and fraction of public Stag actions;
- honest-agent original Stag rate;
- adversarial-agent original Stag rate;
- intended success, $S^{\mathrm{int}}$;
- public success, $S^{\mathrm{pub}}$;
- flip-induced loss: $S^{\mathrm{int}}=1, S^{\mathrm{pub}}=0$;
- flip-induced rescue: $S^{\mathrm{int}}=0, S^{\mathrm{pub}}=1$;
- shared failure: $S^{\mathrm{int}}=0, S^{\mathrm{pub}}=0$;
- shared success: $S^{\mathrm{int}}=1, S^{\mathrm{pub}}=1$.

The public failure rate has the exact decomposition

\[
1-S^{\mathrm{pub}}
=
(1-S^{\mathrm{int}})
+ S^{\mathrm{int}}(1-S^{\mathrm{pub}})
- (1-S^{\mathrm{int}})S^{\mathrm{pub}}.
\]

This separates intention-side failures, flip-induced losses, and flip-induced rescues. The phrase "intention-side" should be used instead of claiming that all such failures were caused by corrupted beliefs.

### Main figure

Create a three-panel figure for the focal $N=5,M=3$ setting:

1. **Honest intended cooperation:** honest-agent original Stag rate against liar fraction.
2. **Outcome comparison:** intended success and public success against liar fraction.
3. **Failure decomposition:** stacked rates for shared failure, flip-induced loss, and flip-induced rescue.

Average within configuration cells first and then average across models so that configurations with repeated runs do not receive disproportionate weight. Show 95% run/configuration-clustered bootstrap intervals.

Put the corresponding facets for every sufficiently supported $N,M$ pair in the appendix.

### Preliminary descriptive result

For base runs with $N=5,M=3$, a model-balanced calculation over the existing rounds gives:

| Liar fraction | Honest intended Stag | Intended success | Public success |
|---:|---:|---:|---:|
| 20% | 0.955 | 0.964 | 0.964 |
| 40% | 0.804 | 0.831 | 0.780 |
| 60% | 0.633 | 0.792 | 0.329 |
| 80% | 0.537 | 0.737 | 0.100 |

These values should be recomputed by the final analysis code with the weighting and clustered uncertainty described above. They already indicate the likely central conclusion: honest intentions deteriorate, but the sharp public-outcome collapse at high corruption is mainly produced by the mechanical flip.

### Interpretation

The vertical line at $1-M/N$ should be relabeled as a **mechanical feasibility boundary under all intended-Stag choices**. It should not be presented as independent evidence of an emergent behavioral tipping point.

## Analysis 2: Public history versus hidden original history

### Objective

Test whether honest agents' original choices are more consistent with the corrupted information they actually observe than with the hidden original actions of earlier speakers.

### Per-turn histories

For every honest turn with at least one earlier speaker in the same round, reconstruct:

\[
q^{\mathrm{pub}}_{irt}
=
\frac{\#\text{ prior public Stag actions}}
{\#\text{ prior speakers}}
\]

and

\[
q^{\mathrm{orig}}_{irt}
=
\frac{\#\text{ prior original Stag choices}}
{\#\text{ prior speakers}}.
\]

The first is available to the agent. The second is an analyst-only oracle that removes the earlier speakers' final mechanical flips.

For each history, calculate the threshold-implied action using the existing $q^*$ function. Compare the implied action with the honest agent's original action.

### Evaluation

Report, on identical observations:

- threshold-rule match rate;
- Brier score for binned empirical response curves;
- log loss if a fitted probabilistic response model is used;
- paired differences between the public-history and original-history predictors;
- confidence intervals from a bootstrap clustered by run or configuration/seed.

The comparison should be stratified by liar fraction and model. It should also be repeated with the carryover benchmark so that prior rounds are represented, but the within-round benchmark should remain the simplest main-text result.

### Preliminary descriptive result

For $N=5,M=3$, model-balanced threshold-rule match rates at high corruption are approximately:

| Number of liars | Public-history match | Original-history match |
|---:|---:|---:|
| 3 | 0.659 | 0.460 |
| 4 | 0.713 | 0.495 |

If this difference remains under clustered inference, it is direct behavioral evidence that honest decisions track the corrupted public transcript rather than the hidden pre-flip choices.

### Interpretation constraint

This is evidence of transcript-consistent behavior, not direct recovery of latent beliefs. The paper should describe these quantities as public-report response benchmarks.

## Analysis 3: Sequential exposure to corrupted speakers

### Objective

Use speaking-order variation to show when the behavioral effect emerges.

### Derived per-turn variables

For each honest turn, reconstruct:

- number of prior speakers;
- number of prior adversarial speakers;
- fraction of prior speakers that are adversarial;
- number and fraction of prior public Stag actions;
- number and fraction of prior original Stag choices;
- round and speaking position;
- previous-round public Stag rate;
- the agent's previous-round original choice, where available.

### Negative control

The first honest speaker in round 1 has observed no other reports, and the prompt does not disclose the number of adversarial agents. Conditional on model, $N$, $M$, and other run parameters, its original Stag rate should not systematically vary with liar exposure.

The substantive effect should appear among later honest speakers who have already seen corrupted messages. Present the first-speaker comparison as a negative-control diagnostic, with sample sizes shown because high-liar settings contain fewer honest first speakers.

### Response model

Fit an honest-action model such as

\[
\Pr(a_{irt}=1)
=
\operatorname{logit}^{-1}
\left(
\beta q^{\mathrm{pub}}_{irt}
+ \gamma E_{irt}
+ \delta_{\mathrm{model},N,M,\mathrm{round},\mathrm{position}}
\right),
\]

where $E_{irt}$ is prior exposure to adversarial speakers. Because public Stag share partly mediates adversarial exposure, estimate the following nested specifications:

1. adversarial exposure plus structural controls;
2. public Stag share plus structural controls;
3. both adversarial exposure and public Stag share plus controls.

Report average marginal effects rather than raw log-odds. Cluster uncertainty by run. Treat attenuation across models as a descriptive pathway analysis, not a fully identified causal mediation effect.

### Figure

Plot honest original Stag probability against the prior public Stag share, with separate curves for round 1 and later rounds. Add a second panel showing the estimated effect of prior adversarial exposure by speaking position.

This analysis should replace or substantially qualify the current "liar influence" metric. The existing fraction of later speakers who choose the same public action is confounded by action prevalence, speaking position, and the final consensus; it is not by itself a causal influence measure.

## Analysis 4: Honest-agent welfare decomposition

### Objective

Express the distinction between decision changes and mechanical changes in payoff units.

### Truthful-implementation payoff

For each logged decision, recompute payoff using:

- the original action $a_{irt}$; and
- the intended-success indicator $S^{\mathrm{int}}_{rt}$.

Call this $U^{\mathrm{truthful}}$. It is the payoff under truthful implementation of the same recorded intentions, not a counterfactual in which the preceding transcript was uncorrupted.

Compare it with the realized payoff $U^{\mathrm{public}}$:

\[
\Delta U^{\mathrm{mechanical}}
=
U^{\mathrm{truthful}}-U^{\mathrm{public}}.
\]

Report honest-agent means by liar fraction and model.

### Preliminary descriptive result

For $N=5,M=3$, the model-balanced endpoint comparison is approximately:

| Liar fraction | Realized honest payoff | Truthful payoff with same intentions | Mechanical gap |
|---:|---:|---:|---:|
| 20% | 3.909 | 3.909 | 0.000 |
| 80% | 1.292 | 3.073 | 1.782 |

This analysis translates the mechanism decomposition into an interpretable system cost.

## Analysis 5: Reanalysis of the random-noise ablation

### Objective

Use the existing `b3` data to distinguish deterministic anti-correlation from generic report noise.

For the existing matched base/`b3` configurations, separately compare:

- honest original Stag rate;
- intended success;
- public success;
- flip-induced loss and rescue rates;
- truthful and realized honest payoffs.

Conduct paired tests at the run-round level and retain Holm correction across rounds. The existing public-success comparison indicates a round-1 advantage of roughly 19.2 percentage points for random noise. The new decomposition should establish whether that difference comes mainly from original behavior, the public action mapping, or both.

This is a robustness analysis and should not replace the main intended/public decomposition.

## Statistical treatment and data weighting

### Analysis population

The main analyses should use default runs only:

- `order_ablation == "a1"`;
- `adversary_ablation == "base"`;
- `heterogeneity_ablation == "h1"`.

Ablations should be analyzed separately.

### Units of analysis

- Use run-rounds for success and payoff outcomes.
- Use agent turns for action-response models, but cluster uncertainty by run.
- Do not use ordinary Bernoulli standard errors that treat all agent turns as independent.

### Configuration weighting

The collected data are unbalanced. Most model/configuration cells contain one run, while a subset contains repeated stochastic runs, often with the same simulation seed. To avoid overweighting those cells:

1. average repeated runs within `model x N x M x F x T x seed` cells;
2. average cell estimates within model;
3. average model estimates for the main cross-model result;
4. retain model-specific results in appendix facets;
5. bootstrap configuration/seed cells within model.

Show raw observation counts and the number of independent configuration/seed cells in every table.

### Common support

The protocol description must match the logs. The base data use a common grid close to:

- $N=2$, $F\in\{0,1\}$;
- $N=3$, $F\in\{1,2\}$;
- $N=5$, $F\in\{1,2,3,4\}$;
- $T\in\{1,2,4\}$;
- thresholds including some $M=N$ settings.

There is a single $N=4,M=4$ base outlier. It should be excluded from general grid claims or reported separately. In particular, the paper must not describe the data as a complete $N=2,\ldots,5$, $F=0,\ldots,N-1$, $M=1,\ldots,N-1$, $T=1,\ldots,4$ factorial design.

Because there are no fully honest $F=0$ runs for $N=3$ or $N=5$, the effect of corruption relative to a fully honest baseline is not identified for those group sizes. Main claims should compare observed liar levels or use sequential within-run exposure, rather than treating the 0% aggregate as a universal baseline.

## Required changes to the paper's interpretation

### Methods

- Explicitly distinguish original choice, public action, and executed outcome.
- State that payoffs and success are calculated from public actions.
- State that adversarial roles are programmatic channel interventions; the LLM is not instructed to act strategically as an adversary.
- Explain that later original choices may already respond to earlier public corruption.
- Disclose that the action and justification are changed after the original response, while the original confidence is retained.

### Belief model

The current implementation calculates $q=K/n$. It returns the reliability parameter $\alpha=(N-F)/N$ but does not use it to decontaminate $q$, and agents are not told $F$ in the system prompt. Therefore:

- remove claims about an implemented Bayesian corruption correction;
- describe $q$ as an empirical public-report rate;
- remove the Appendix claim that corruption affects outcomes exclusively through belief formation;
- call threshold comparisons behavioral benchmarks rather than latent-belief measurements.

An analyst-only corruption-aware or original-history benchmark may be included as a contrast, but it must not be described as the belief available to the agent.

### Results

Recommended order:

1. intended/public action and outcome decomposition;
2. public-history versus original-history prediction;
3. sequential exposure and carryover dynamics;
4. honest welfare decomposition;
5. model heterogeneity and random-noise robustness;
6. confidence/calibration as a secondary result.

The current calibration and heterogeneity analyses can be shortened to make room for the mechanism results. Avoid causal language around the existing influence metric.

### Terminology

Preferred terms:

- "original" or "pre-flip choice" instead of "true belief";
- "public" or "executed action" instead of treating it as the only action;
- "corrupted reporter" or "programmatically corrupted agent" where appropriate;
- "ex-post action-outcome alignment" instead of "accuracy";
- "public-report response benchmark" instead of "recovered belief";
- "mechanical feasibility boundary" instead of "theoretical tipping threshold."

## Reproducibility appendix

Include the following verbatim:

1. full agent system prompt;
2. structured JSON output schema;
3. GameMaster round-start message;
4. lie-justification rewrite prompt;
5. public transcript serialization format;
6. transcript-patching procedure;
7. action-flipping rule for base and `b3`;
8. statement that confidence is retained after flipping;
9. temperature and other decoding parameters;
10. model/provider identifiers and dates;
11. random role-assignment and speaking-order procedures;
12. exact realized parameter-grid coverage and run counts.

## Planned outputs

Suggested artifacts are:

- `fig_mechanism_decomposition`: honest intentions, intended/public success, and failure channels;
- `fig_public_vs_original_history`: paired benchmark response curves or match-rate differences;
- `fig_sequential_exposure`: honest action response by prior public history and adversarial exposure;
- `table_welfare_decomposition`: realized versus truthful-same-intentions payoff;
- `table_b3_channel_decomposition`: matched base versus random-noise results;
- `table_data_coverage`: exact run counts and common support;
- `table_clustered_robustness`: estimates under run/configuration clustered inference.

Each output should include the analysis population, aggregation order, uncertainty method, and observation/cell counts in its caption or table note.

## Implementation order

1. Build one canonical enriched agent-turn table from the run and agent-metrics CSVs.
2. Validate that original and public actions coincide for honest agents and are inverted for base-condition corrupted agents.
3. Build the run-round action-channel table.
4. Produce the focal mechanism-decomposition figure and table.
5. Reconstruct public and original pre-turn histories.
6. Produce the paired history-prediction analysis.
7. Fit the sequential-exposure models and negative-control check.
8. Compute truthful-implementation payoffs.
9. Reanalyze matched `b3` runs.
10. Add clustered/bootstrap uncertainty and sensitivity to weighting choices.
11. Generate the exact data-coverage table.
12. Rewrite the Methods, main Results, Abstract, and reproducibility appendix around the decomposed mechanism.

## Completion criteria

The revised analysis is complete when:

- internal choices, public actions, and outcomes are reported separately;
- the share of failures due to each channel is quantified;
- honest action responses are compared against both public and hidden original histories;
- sequential exposure is analyzed with appropriate controls and clustered uncertainty;
- the parameter-grid description matches the actual logs;
- the paper no longer attributes the mechanical boundary solely to belief updating;
- prompts and implementation details are included in the appendix;
- every main result can be regenerated from `logs/all_combination` without new model calls.
