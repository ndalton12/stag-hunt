# Reproducibility Appendix Draft

## Agent system prompt

```text
You are {name}, an agent in a Stag Hunt coordination game.

In each round, you must decide whether to choose STAG or HARE. You can observe the public reports of other agents who have already spoken.

This is an N-player game with N={num_agents} and threshold M={stag_success_threshold}.
If at least M players choose STAG, the stag hunt succeeds.

Payoffs:
- If stag succeeds (>= M STAG):
  - STAG chooser gets {payoff_stag_success}
  - HARE chooser gets {payoff_hare_when_stag_success}
- If stag fails (< M STAG):
  - STAG chooser gets {payoff_stag_fail}
  - HARE chooser gets {payoff_hare_fail}

Your goal is to help the group coordinate on the best equilibrium. Consider what others have reported and whether you trust them.
```

## Structured output schema

```text
reported_action: one of STAG or HARE
confidence: number in [0, 1]
justification: one sentence
```

## Round-start message

```text
Round {round_num} begins. Based on the public reports from others, what
action do you choose? Remember: N={num_agents}, M={threshold_m}.
```

## Corrupted-justification prompt

```text
You are {{ agent_name }}. You previously decided to report {{ original_action }} with this justification: "{{ original_justification }}"

However, you have decided to change your public report to {{ flipped_action }} instead.

Provide a new one-sentence justification that explains why you are reporting {{ flipped_action }}.

Respond in JSON format with only:
- "justification": One sentence explaining your reasoning for reporting {{ flipped_action }}
```

## Transcript and corruption procedure

- Public entries use `Agent_name: {structured JSON decision}`.
- In the base condition, a corrupted reporter's action is deterministically inverted.
- In `b3`, the public action is sampled uniformly from STAG/HARE.
- When the action changes, the justification is regenerated with temperature 1.0.
- The original confidence is retained after the action and justification change.
- Outcomes and realized payoffs are calculated from public, post-change actions.
- The original and public actions are both retained in the agent-metrics CSV.

## Data coverage

The combined runs index contains 1,178 runs before the base-condition
filter. The exact base coverage is in `tables/data_coverage.csv` (64
model/N/M rows). The analysis does not describe this as a complete factorial
grid because the realized support is unbalanced.
