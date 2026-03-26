"""Bayesian belief model for the Stag Hunt simulation.

Implements the mathematical framework from the Methods section (§3–§3.4).

§3   – Agent belief  q^t_i = P(randomly selected opponent plays Stag)
§3.1 – Belief update with linear α-correction for signal corruption
§3.2 – Expected payoff and cutoff threshold q*
§3.3 – Revealed belief inference in round 1 (belief unobservable; action reveals region)
§3.4 – Comparative statics: how q* shifts with M, R_S, H, F
"""

from __future__ import annotations

from scipy.stats import binom

# ============================================================================
# §3.1  Signal reliability and belief update
# ============================================================================


def compute_alpha(num_agents: int, num_liars: int) -> float:
    """Fraction of honest agents — signal reliability α = (N − F) / N.

    A randomly observed signal reflects the sender's true intended action
    with probability α, and is adversarially flipped with probability 1 − α.
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be > 0")
    return (num_agents - num_liars) / num_agents


def compute_belief(
    k_stag: int,
    n_observed: int,
    num_agents: int,
    num_liars: int,
) -> tuple[float, float | None]:
    """Compute the public-belief benchmark for one agent (§3.1).

    When n_observed == 0 (first agent to speak in round 1), there are no
    prior reports to update from.  In that case q_hat is returned as None —
    the belief is *unobservable*, not defaulted to 0.5.
    Use infer_first_round_belief() to characterise round-1 beliefs instead.

    Args:
        k_stag:     Number of public STAG reports seen before this agent speaks.
        n_observed: Total public reports observed before this agent speaks.
        num_agents: N — total agents in the game.
        num_liars:  F — number of adversarial agents.

    Returns:
        (alpha, q_hat)
        alpha       — signal reliability (N−F)/N; always returned.
        q_hat       — empirical public STAG rate K / n_observed over the
                      reports currently in the agent's information set;
                      None when n_observed == 0.
    """
    alpha = compute_alpha(num_agents, num_liars)

    if n_observed == 0:
        # No reports seen yet — belief is unobservable in round 1.
        # Caller should use infer_first_round_belief() for the §3.3 inference.
        return alpha, None

    q_hat = k_stag / n_observed

    return alpha, q_hat


# ============================================================================
# §3.2  Expected payoff and belief threshold q*
# ============================================================================


def expected_payoff_stag(
    q: float,
    num_agents: int,
    threshold_m: int,
    payoff_stag_success: float,
    payoff_stag_fail: float,
) -> float:
    """Expected payoff from choosing STAG given belief q (§3.2).

    E[U_S] = R_S · P(K ≥ M−1) + R_F · P(K < M−1)

    where K ~ Binomial(N−1, q) counts how many *other* agents choose Stag.
    Agent i needs at least M−1 others so the total reaches threshold M.
    """
    if threshold_m <= 1:
        p_success = 1.0
    else:
        p_success = float(1 - binom.cdf(threshold_m - 2, num_agents - 1, q))
    return payoff_stag_success * p_success + payoff_stag_fail * (1 - p_success)


def compute_q_star(
    num_agents: int,
    threshold_m: int,
    payoff_stag_success: float,
    payoff_stag_fail: float,
    payoff_hare_safe: float,
    tol: float = 1e-7,
) -> float:
    """Find the rational cooperation threshold q* via bisection (§3.2).

    q* is defined implicitly by  E[U_S(q*)] = H  (H = safe Hare payoff).

    This is a *game-level constant*: it depends only on the fixed structural
    parameters (N, M, payoffs) and does not change across rounds.  Callers
    should compute it once at simulation initialisation, not per round.

    Returns q* ∈ [0.0, 1.0].
      • q* = 0.0  — Stag is dominant even at the lowest possible belief.
      • q* = 1.0  — Hare always dominates.
    """

    def f(q: float) -> float:
        return (
            expected_payoff_stag(
                q, num_agents, threshold_m, payoff_stag_success, payoff_stag_fail
            )
            - payoff_hare_safe
        )

    if f(0.0) >= 0:
        return 0.0
    if f(1.0) < 0:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


def compute_rational_action(
    q: float,
    num_agents: int,
    threshold_m: int,
    payoff_stag_success: float,
    payoff_stag_fail: float,
    payoff_hare_safe: float,
) -> str:
    """Best-response action under the cutoff strategy (§3.2).

    Returns "STAG" if E[U_S(q)] ≥ H, else "HARE".
    """
    e_stag = expected_payoff_stag(
        q, num_agents, threshold_m, payoff_stag_success, payoff_stag_fail
    )
    return "STAG" if e_stag >= payoff_hare_safe else "HARE"


# ============================================================================
# §3.3  Revealed belief inference (round 1 only)
# ============================================================================


def infer_first_round_belief(action: str, q_star: float) -> dict:
    """Characterise the agent's unobservable prior belief from its round-1 action (§3.3).

    In round 1 there are no prior reports, so q^0_i cannot be computed
    from observations.  However, the cutoff strategy implies:
        STAG chosen  →  q^0_i ∈ [q*, 1.0]
        HARE chosen  →  q^0_i ∈ [0.0, q*)

    This is a *set identification*, not a point estimate.  It tells us which
    side of the threshold the latent belief falls on, nothing more.

    Returns a dict with the inferred belief region and the q* used.
    """
    if action == "STAG":
        region = f"[{q_star:.4f}, 1.0]  (q⁰ᵢ ≥ q*)"
    else:
        region = f"[0.0, {q_star:.4f})  (q⁰ᵢ < q*)"

    return {
        "action": action,
        "q_star_used": q_star,
        "inferred_belief_region": region,
    }


# ============================================================================
# §3.4  Comparative statics helpers
# ============================================================================


def q_star_sensitivity(
    num_agents: int,
    threshold_m: int,
    payoff_stag_success: float,
    payoff_stag_fail: float,
    payoff_hare_safe: float,
) -> dict:
    """Estimate the sign of ∂q*/∂X for each structural parameter (§3.4).

    Expected signs from the paper:
        ∂q*/∂M  > 0  (stricter threshold → cooperation harder)
        ∂q*/∂RS < 0  (higher success payoff → cooperation easier)
        ∂q*/∂H  > 0  (higher safe Hare payoff → cooperation harder)

    Returns a dict with q_star and the sign (+1 / -1 / 0) for each parameter.
    Signs are estimated via finite differences.
    """
    eps = 0.01
    q0 = compute_q_star(
        num_agents, threshold_m, payoff_stag_success, payoff_stag_fail, payoff_hare_safe
    )

    dM_sign: int | None = None
    if threshold_m < num_agents:
        q_m = compute_q_star(
            num_agents,
            threshold_m + 1,
            payoff_stag_success,
            payoff_stag_fail,
            payoff_hare_safe,
        )
        dM_sign = 1 if q_m > q0 else (-1 if q_m < q0 else 0)

    q_rs = compute_q_star(
        num_agents,
        threshold_m,
        payoff_stag_success + eps,
        payoff_stag_fail,
        payoff_hare_safe,
    )
    dRS_sign = 1 if q_rs > q0 else (-1 if q_rs < q0 else 0)

    q_h = compute_q_star(
        num_agents,
        threshold_m,
        payoff_stag_success,
        payoff_stag_fail,
        payoff_hare_safe + eps,
    )
    dH_sign = 1 if q_h > q0 else (-1 if q_h < q0 else 0)

    return {
        "q_star": q0,
        "dq_star_dM_sign": dM_sign,  # expected +1
        "dq_star_dRS_sign": dRS_sign,  # expected -1
        "dq_star_dH_sign": dH_sign,  # expected +1
    }
