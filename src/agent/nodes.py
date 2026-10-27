"""LangGraph node functions.

Each node takes the current AgentState plus `deps` (bound in graph.py) and
returns a partial state update. LLM nodes go through `decide()`, which
parses, semantically validates, and allows one bounded repair. Python then
executes the validated decision with the predefined tools in tools.py.

Only `final_evaluation_node` receives the held-out test partition.
"""

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import tools as T
from .errors import TOOL_ERROR, DecisionError
from .llm import decide
from .metrics import METRICS, direction, meets_threshold, supported_metrics
from .selection import decide_route, select_best
from .state import AnalysisPlan, Critique, FeaturePlan, ModelChoice, RunConfig
from .validation import (
    HYPERPARAM_BOUNDS,
    models_for_task,
    validate_feature_plan,
    validate_model_choice,
    validate_plan,
)


@dataclass
class Deps:
    llm: object
    config: RunConfig
    dataset: object  # DatasetBundle (its df is not used inside the graph)
    run_id: str
    output_dir: str


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _node_event(node: str, started: float, status: str, candidate_id=None, **extra) -> dict:
    event = {
        "node": node,
        "candidate_id": candidate_id,
        "type": "node",
        "status": status,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    event.update(extra)
    return event


def _n_classes(dev_df: pd.DataFrame, deps: Deps) -> Optional[int]:
    if deps.dataset.task_type != "classification":
        return None
    return int(dev_df[deps.dataset.target_column].nunique())


def _next_candidate_id(state) -> str:
    return f"C{len(state.get('candidates', [])) + 1}"


def _latest_feedback(state) -> str:
    critiques = state.get("critiques") or []
    if not critiques:
        return "None yet - this is the first attempt."
    last = critiques[-1]
    return last.get("feedback") or "The previous critic call failed; no feedback available."


def _history_lines(candidates: list[dict], metric: str) -> str:
    """Validation-only summary of earlier attempts, for prompts."""
    if not candidates:
        return "No earlier candidates."
    lines = []
    for c in candidates:
        if c["status"] == "valid":
            p = c["validation"]["primary"]
            model = c["model_choice"]
            lines.append(
                f"- {c['candidate_id']}: {model['model_name']} "
                f"(n_estimators={model['n_estimators']}, max_depth={model['max_depth']}), "
                f"features: {_feature_summary(c['feature_plan'])}; "
                f"validation {metric} = {p['mean']:.4f} ± {p['std']:.4f}"
            )
        else:
            lines.append(
                f"- {c['candidate_id']}: FAILED at {c['failure']['stage']} "
                f"({c['failure']['kind']}): {'; '.join(c['failure']['messages'])[:300]}"
            )
    return "\n".join(lines)


def _feature_summary(feature_plan: dict) -> str:
    counts: dict[str, int] = {}
    for step in feature_plan["steps"]:
        counts[step["transform"]] = counts.get(step["transform"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def _failed_candidate(cid, stage, kind, messages, feature_plan=None, model_choice=None) -> dict:
    return {
        "candidate_id": cid,
        "status": "failed",
        "feature_plan": feature_plan.model_dump() if feature_plan else None,
        "model_choice": model_choice.model_dump() if model_choice else None,
        "validation": None,
        "warnings": [],
        "failure": {"stage": stage, "kind": kind, "messages": list(messages)},
    }


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def plan_node(state, deps: Deps):
    started = time.perf_counter()
    ds, cfg = deps.dataset, deps.config
    dev = state["dev_df"]
    n_classes = _n_classes(dev, deps)
    allowed = supported_metrics(ds.task_type, n_classes)
    features = [c for c in dev.columns if c != ds.target_column]
    metric_line = (
        f"The user fixed the primary metric to '{cfg.primary_metric}'; set recommended_metric to it."
        if cfg.primary_metric
        else f"recommended_metric must be one of: {allowed}."
    )
    prompt = f"""You are planning a small, bounded tabular modeling run.

Dataset: {ds.name}
Task type: {ds.task_type}
Target column: {ds.target_column}
Description: {ds.description}
Feature columns ({len(features)}): {features}

Write a 2-4 sentence analysis plan. {metric_line}
Do not propose an acceptance threshold; success criteria come from the user."""

    try:
        plan, events = decide(
            deps.llm, "plan", AnalysisPlan, prompt,
            lambda p: validate_plan(p, ds.task_type, n_classes, cfg.primary_metric),
            cfg.max_repair_attempts,
        )
    except DecisionError as err:
        return {
            "fatal_error": {"node": "plan", "kind": err.kind, "messages": err.messages},
            "stop_reason": "failed",
            "events": err.events + [_node_event("plan", started, err.kind)],
        }

    metric = cfg.primary_metric or plan.recommended_metric
    source = "user" if cfg.primary_metric else "planner"
    return {
        "plan": plan,
        "primary_metric": metric,
        "metric_source": source,
        "events": events + [_node_event("plan", started, "ok", primary_metric=metric, metric_source=source)],
    }


def eda_node(state, deps: Deps):
    started = time.perf_counter()
    try:
        summary, figures = T.run_eda(
            state["dev_df"], deps.dataset.target_column, f"{deps.output_dir}/figures"
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "fatal_error": {"node": "eda", "kind": TOOL_ERROR, "messages": [message]},
            "stop_reason": "failed",
            "events": [_node_event("eda", started, TOOL_ERROR, error=message)],
        }
    return {
        "eda_summary": summary,
        "eda_figures": figures,
        "events": [_node_event("eda", started, "ok", n_rows=summary["n_rows"], partition="development")],
    }


def feature_engineering_node(state, deps: Deps):
    started = time.perf_counter()
    ds, cfg = deps.dataset, deps.config
    cid = _next_candidate_id(state)
    dev = state["dev_df"]
    stats = state["eda_summary"]["features"]
    feature_lines = "\n".join(
        f"- {c} ({s['dtype']}): min={s['min']}, max={s['max']}, skew={s['skew']}"
        for c, s in stats.items()
    )
    prompt = f"""You are choosing preprocessing for a {ds.task_type} model (candidate {cid}).

Target column (never include it): {ds.target_column}
Input feature columns, computed on the development partition:
{feature_lines}

Contract: return EXACTLY ONE step for EVERY feature listed above - no
missing columns, no duplicates, no other column names. Allowed transforms:
- scale: standardize (fit on training folds only)
- log1p: log(1 + x), only for columns whose min is >= 0
- keep: pass through unchanged
- drop: exclude from the model (at least one feature must remain)

Earlier candidates (validation scores only):
{_history_lines(state.get('candidates', []), state['primary_metric'])}

Critic feedback on the last attempt: {_latest_feedback(state)}"""

    try:
        feature_plan, events = decide(
            deps.llm, "feature_engineering", FeaturePlan, prompt,
            lambda fp: validate_feature_plan(fp, dev, ds.target_column),
            cfg.max_repair_attempts, candidate_id=cid,
        )
    except DecisionError as err:
        return {
            "feature_stage_failed": True,
            "candidates": [_failed_candidate(cid, "feature_engineering", err.kind, err.messages)],
            "events": err.events + [_node_event("feature_engineering", started, err.kind, cid)],
        }
    return {
        "feature_stage_failed": False,
        "current_feature_plan": feature_plan,
        "feature_plan_origin": cid,
        "events": events + [_node_event("feature_engineering", started, "ok", cid,
                                        transforms=_feature_summary(feature_plan.model_dump()))],
    }


def modeling_node(state, deps: Deps):
    started = time.perf_counter()
    ds, cfg = deps.dataset, deps.config
    cid = _next_candidate_id(state)
    metric = state["primary_metric"]
    feature_plan: FeaturePlan = state["current_feature_plan"]
    allowed = models_for_task(ds.task_type)
    bounds = "\n".join(
        f"- {m}: " + ", ".join(f"{p} in [{lo}, {hi}] or null" for p, (lo, hi, _) in HYPERPARAM_BOUNDS[m].items())
        if m in HYPERPARAM_BOUNDS else f"- {m}: no hyperparameters (set both to null)"
        for m in allowed
    )
    prompt = f"""Choose a scikit-learn model for this {ds.task_type} task (candidate {cid}).

Allowed models and hyperparameter bounds:
{bounds}

Primary metric: {metric} ({direction(METRICS[metric])}), measured by
{cfg.cv_folds}-fold cross-validation on the development partition.
Feature plan in use: {_feature_summary(feature_plan.model_dump())}

Earlier candidates (validation scores only):
{_history_lines(state.get('candidates', []), metric)}

Critic feedback on the last attempt: {_latest_feedback(state)}"""

    feature_dump = feature_plan.model_dump()
    origin = state.get("feature_plan_origin")
    try:
        choice, events = decide(
            deps.llm, "modeling", ModelChoice, prompt,
            lambda mc: validate_model_choice(mc, ds.task_type),
            cfg.max_repair_attempts, candidate_id=cid,
        )
    except DecisionError as err:
        cand = _failed_candidate(cid, "modeling", err.kind, err.messages, feature_plan)
        cand["feature_plan_origin"] = origin
        return {"candidates": [cand], "events": err.events + [_node_event("modeling", started, err.kind, cid)]}

    eval_started = time.perf_counter()
    try:
        result = T.evaluate_candidate(
            state["dev_df"], ds.target_column, ds.task_type, feature_plan, choice,
            metric, cfg.cv_folds, cfg.seed,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        cand = _failed_candidate(cid, "evaluation", TOOL_ERROR, [message], feature_plan, choice)
        cand["feature_plan_origin"] = origin
        return {
            "candidates": [cand],
            "events": events + [_node_event("modeling", started, TOOL_ERROR, cid, error=message)],
        }

    candidate = {
        "candidate_id": cid,
        "status": "valid",
        "feature_plan": feature_dump,
        "feature_plan_origin": origin,
        "model_choice": choice.model_dump(),
        "validation": {k: result[k] for k in ("primary_metric", "primary", "metrics", "n_folds")},
        "warnings": result["warnings"],
        "failure": None,
    }
    return {
        "candidates": [candidate],
        "events": events + [_node_event(
            "modeling", started, "ok", cid,
            model=choice.model_name,
            cv_seconds=round(time.perf_counter() - eval_started, 3),
            validation_primary=round(result["primary"]["mean"], 6),
        )],
    }


def critic_node(state, deps: Deps):
    started = time.perf_counter()
    cfg = deps.config
    candidates = state["candidates"]
    metric = state["primary_metric"]
    last = candidates[-1]
    target_line = (
        f"User acceptance target: validation {metric} "
        f"{'>=' if METRICS[metric].greater_is_better else '<='} {cfg.acceptance_threshold}."
        if cfg.acceptance_threshold is not None
        else "No acceptance target was specified by the user."
    )
    prompt = f"""You are reviewing candidate {last['candidate_id']} in a bounded modeling run.

Primary metric: {metric} ({direction(METRICS[metric])}). All scores below are
cross-validation results on the development partition.
{target_line}
Attempts used: {len(candidates)} of {cfg.max_candidates}.

Candidate history:
{_history_lines(candidates, metric)}

Recommend one action:
- iterate_features: revise preprocessing for the next attempt
- iterate_model: keep the current preprocessing and change the model
- proceed: stop iterating and report the best candidate so far
Give brief, specific feedback for the next attempt. The application, not
you, enforces the attempt budget and any acceptance target."""

    critique, error, events = None, None, []
    try:
        critique, events = decide(
            deps.llm, "critic", Critique, prompt, lambda c: [], cfg.max_repair_attempts,
            candidate_id=last["candidate_id"],
        )
    except DecisionError as err:
        error, events = {"kind": err.kind, "messages": err.messages}, err.events

    action = critique.recommended_action if critique else None
    route, stop_reason, override = decide_route(
        candidates, metric, cfg.acceptance_threshold, cfg.max_candidates, action,
        has_feature_plan=state.get("current_feature_plan") is not None,
    )
    best = select_best(candidates, metric)
    record = {
        "candidate_id": last["candidate_id"],
        "recommended_action": action,
        "feedback": critique.feedback if critique else None,
        "route": route,
        "overridden": override is not None,
        "override_reason": override,
        "error": error,
    }
    update = {
        "critiques": [record],
        "next_step": route,
        "best_candidate_id": best["candidate_id"] if best else None,
        "events": events + [_node_event(
            "critic", started, "ok" if critique else error["kind"], last["candidate_id"],
            recommended_action=action, route=route, override_reason=override,
            stop_reason=stop_reason, best_candidate_id=best["candidate_id"] if best else None,
        )],
    }
    if route == "stop":
        update["stop_reason"] = stop_reason
    return update


def final_evaluation_node(state, deps: Deps, holdout: pd.DataFrame):
    """Runs once, after selection. Its output never flows back into the loop."""
    started = time.perf_counter()
    ds, cfg = deps.dataset, deps.config
    best_id = state.get("best_candidate_id")
    best = next((c for c in state.get("candidates", []) if c["candidate_id"] == best_id), None)
    if best is None:
        return {
            "final_test": None,
            "events": [_node_event("final_evaluation", started, "skipped",
                                   reason="no valid candidate to evaluate")],
        }
    try:
        result = T.final_evaluate(
            state["dev_df"], holdout, ds.target_column, ds.task_type,
            FeaturePlan.model_validate(best["feature_plan"]),
            ModelChoice.model_validate(best["model_choice"]),
            cfg.seed,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        return {
            "final_test": {"candidate_id": best_id, "error": {"kind": TOOL_ERROR, "messages": [message]}},
            "events": [_node_event("final_evaluation", started, TOOL_ERROR, best_id, error=message)],
        }

    metric = state["primary_metric"]
    meets = None
    if cfg.acceptance_threshold is not None:
        meets = meets_threshold(result["metrics"][metric], cfg.acceptance_threshold, METRICS[metric])
    final_test = {"candidate_id": best_id, "meets_threshold": meets, **result}
    return {
        "final_test": final_test,
        "events": [_node_event("final_evaluation", started, "ok", best_id,
                               partition="final_test", test_primary=round(result["metrics"][metric], 6))],
    }


def report_node(state, deps: Deps):
    from ..report.generator import generate_narrative, write_reports

    started = time.perf_counter()
    narrative, events = generate_narrative(state, deps)
    state = {**state, "narrative": narrative, "events": state.get("events", []) + events}
    paths = write_reports(state, deps)
    return {
        "narrative": narrative,
        "report_paths": paths,
        "events": events + [_node_event("report", started, "ok")],
    }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def route_if_fatal(state):
    return "report" if state.get("fatal_error") else "continue"


def route_after_features(state):
    return "critic" if state.get("feature_stage_failed") else "modeling"


def route_after_critic(state):
    return state["next_step"]
