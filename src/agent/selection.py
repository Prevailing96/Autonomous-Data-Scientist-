"""Deterministic candidate selection, stopping rules, and acceptance status.

These are plain functions of the candidate history and the run config, so
the rules that decide what ships live in code, not in the critic's prompt.
"""

from typing import Optional

from .metrics import METRICS, is_better, meets_threshold

# How close two CV means must be to count as a tie (absorbs float noise).
TIE_DECIMALS = 10

STOP_REASONS = {
    "target_met": "The best validation score reached the user's acceptance threshold.",
    "critic_proceeded": "The critic recommended stopping; the best validated candidate was selected.",
    "budget_exhausted": "The candidate budget was reached; the best validated candidate was selected.",
    "critic_failed": "The critic could not return a valid decision, so the run stopped "
    "with the best candidate so far.",
    "failed": "The run could not produce any valid candidate.",
}


def valid_candidates(candidates: list[dict]) -> list[dict]:
    return [c for c in candidates if c.get("status") == "valid"]


def select_best(candidates: list[dict], metric: str) -> Optional[dict]:
    """Best valid candidate by primary CV mean.

    Tie-breaker, in order: lower CV standard deviation (more stable), then
    the earlier candidate (fewer adaptive decisions behind it).
    """
    spec = METRICS[metric]
    best = None
    for cand in valid_candidates(candidates):
        if best is None:
            best = cand
            continue
        a = round(cand["validation"]["primary"]["mean"], TIE_DECIMALS)
        b = round(best["validation"]["primary"]["mean"], TIE_DECIMALS)
        if is_better(a, b, spec):
            best = cand
        elif a == b and cand["validation"]["primary"]["std"] < best["validation"]["primary"]["std"]:
            best = cand
        # Exact ties on both keep the earlier candidate.
    return best


def decide_route(
    candidates: list[dict],
    metric: str,
    threshold: Optional[float],
    max_candidates: int,
    critic_action: Optional[str],
    has_feature_plan: bool,
) -> tuple[str, Optional[str], Optional[str]]:
    """Apply the stopping rules to the critic's recommendation.

    Returns (route, stop_reason, override_note). `route` is one of
    iterate_features / iterate_model / stop. `override_note` explains why the
    code did not follow the critic, or is None if it did.
    """
    best = select_best(candidates, metric)
    attempts = len(candidates)

    if best is not None and threshold is not None:
        if meets_threshold(best["validation"]["primary"]["mean"], threshold, METRICS[metric]):
            note = None if critic_action == "proceed" else "acceptance threshold already met"
            return "stop", "target_met", note

    if critic_action is None:
        return "stop", ("critic_failed" if best else "failed"), "critic unavailable"

    if critic_action == "proceed":
        if best is not None:
            return "stop", "critic_proceeded", None
        if attempts < max_candidates:
            return "iterate_features", None, "cannot proceed without a valid candidate"
        return "stop", "failed", "no valid candidate and budget exhausted"

    if attempts >= max_candidates:
        reason = "budget_exhausted" if best else "failed"
        return "stop", reason, f"candidate budget of {max_candidates} used"

    if critic_action == "iterate_model" and not has_feature_plan:
        return "iterate_features", None, "no valid feature plan exists to reuse"

    return critic_action, None, None


def acceptance_status(best: Optional[dict], metric: Optional[str], threshold: Optional[float]) -> str:
    """Validation-based acceptance: met / not_met / not_specified."""
    if threshold is None:
        return "not_specified"
    if best is None or metric is None:
        return "not_met"
    met = meets_threshold(best["validation"]["primary"]["mean"], threshold, METRICS[metric])
    return "met" if met else "not_met"
