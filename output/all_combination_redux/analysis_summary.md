# Redux Analysis Summary

This directory contains the mechanism-focused reanalysis of
`logs/all_combination`. It was generated entirely from existing CSV logs; no
new model calls or simulations were made.

## Focal mechanism decomposition: N=5, M=3

| liar_share | honest_original_stag_rate | intended_success | public_success | flip_induced_loss |
| --- | --- | --- | --- | --- |
| 20% | 0.963 | 0.972 | 0.972 | 0.000 |
| 40% | 0.836 | 0.869 | 0.790 | 0.083 |
| 60% | 0.645 | 0.841 | 0.266 | 0.611 |
| 80% | 0.516 | 0.778 | 0.123 | 0.659 |

`intended_success` applies the threshold to recorded pre-flip choices.
`public_success` is the executed outcome in the original simulation.
`flip_induced_loss` is the share of run-rounds where intended choices succeed
but public actions fail.

## Public versus hidden-original history

| liar_share | public_history_match | original_history_match |
| --- | --- | --- |
| 20% | 0.913 | 0.980 |
| 40% | 0.646 | 0.809 |
| 60% | 0.645 | 0.472 |
| 80% | 0.698 | 0.476 |

The public-history benchmark uses only reports available to the deciding
agent. The hidden-original benchmark is an analyst-only contrast.

## Honest welfare decomposition

| liar_share | honest_realized_payoff | honest_truthful_payoff | honest_mechanical_payoff_gap |
| --- | --- | --- | --- |
| 20% | 3.927 | 3.927 | 0.000 |
| 40% | 3.466 | 3.661 | 0.196 |
| 60% | 1.702 | 3.218 | 1.516 |
| 80% | 1.444 | 3.032 | 1.587 |

The truthful payoff applies the payoff rule to the same logged pre-flip
choices. It is not a no-corruption transcript counterfactual.

## Sequential response models

| specification | predictor | estimate | cluster_se | ci_low | ci_high | p_value | n_turns | n_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exposure only | prior_liar_share | -0.120 | 0.048 | -0.213 | -0.026 | 0.012 | 2852 | 497 |
| Public history only | prior_public_stag_share | 0.499 | 0.041 | 0.418 | 0.580 | 0.000 | 2852 | 497 |
| Exposure + public history | prior_liar_share | 0.303 | 0.063 | 0.180 | 0.425 | 0.000 | 2852 | 497 |
| Exposure + public history | prior_public_stag_share | 0.594 | 0.044 | 0.508 | 0.679 | 0.000 | 2852 | 497 |

These are linear probability models with model, game-condition, round, and
speaking-position fixed effects. Standard errors are clustered by
model/configuration/seed.

## Public-history response rules

Figure 14 compares naive within-round aggregation, cross-round carryover, and
trust-weighted public-history estimates. The rules change $\hat{q}$; the game
threshold $q^*$ remains fixed for a given payoff/threshold configuration.

## Random-noise robustness

The matched base-versus-b3 table contains 28 round/metric rows. See
`tables/b3_matched_decomposition.csv` for the complete results.

## Files

- `fig1_mechanism_decomposition`: honest choices and intended versus public success.
- `fig1b_mechanism_by_model`: the same three measures separated by model.
- `fig2_public_vs_original_history`: which action history better matches honest decisions.
- `fig3_sequential_exposure`: response curves and fixed-effect exposure estimates.
- `fig4_welfare_decomposition`: realized versus truthful-implementation honest payoff.
- `fig9_coordination_dynamics_redux`: honest choices and both outcome layers by round.
- `fig10_speaking_position_redux`: descriptive honest Stag choices by fixed position.
- `fig14_public_history_response_redux`: model-specific response under naive, carryover, and trust-weighted public-history rules.
- `figA1_mechanism_facets`: intended/public outcomes across all supported N/M games.
- `tables/`: machine-readable summaries, model estimates, coverage, and robustness checks.
- `reproducibility_appendix.md`: prompts and implementation details to include in the paper appendix.
- `pdf/`: vector versions of all figures.

In Figure 10, a missing position/share point means that no honest agent occupied
that fixed speaking position in the supported runs; it is not a zero-valued
observation or a rendering failure.
