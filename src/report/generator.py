"""Writes a run's Markdown report, self-contained HTML report, and JSON trace.

Every number in the reports is rendered straight from tool outputs in the
state. The LLM narrative is optional and clearly labeled; if it fails, the
deterministic report is still complete. The report is built once as a list
of blocks and rendered to both formats, so they cannot drift apart. All
dynamic HTML content is escaped.
"""

import base64
import html
import json
import os
import re
import time
from datetime import datetime, timezone

from ..agent.metrics import METRICS, direction
from ..agent.selection import STOP_REASONS, acceptance_status, select_best

LIMITATIONS = [
    "Candidate selection adapts to cross-validation results over several attempts, so "
    "the selected candidate's validation score is optimistically biased. The final test "
    "partition is held out and scored once to give an unbiased check.",
    "One random train/test split and one CV seed. With a few hundred rows, the final "
    "test score can move by a few points under a different seed.",
    "The action space is small on purpose: four per-column transforms, six model "
    "families, two bounded hyperparameters. No imputation, encoding, or feature creation.",
    "Reasoning and feedback text explain each decision; the metrics in the tables are "
    "the authoritative record.",
]


# --------------------------------------------------------------------------
# Narrative (optional, LLM-generated)
# --------------------------------------------------------------------------

def generate_narrative(state, deps) -> tuple:
    """Returns (narrative or None, events). Never raises."""
    outcome = build_outcome(state, deps)
    facts = {
        "dataset": deps.dataset.name,
        "description": deps.dataset.description,
        "task_type": deps.dataset.task_type,
        "plan": state["plan"].approach_summary if state.get("plan") else None,
        "primary_metric": state.get("primary_metric"),
        "stop_reason": outcome["stop_reason"],
        "acceptance_status": outcome["acceptance_status"],
        "candidates": [
            {
                "id": c["candidate_id"],
                "status": c["status"],
                "model": (c.get("model_choice") or {}).get("model_name"),
                "failure": (c.get("failure") or {}).get("kind"),
            }
            for c in state.get("candidates", [])
        ],
        "selected_candidate": outcome["selected_candidate_id"],
        "critic_feedback": [c.get("feedback") for c in state.get("critiques", []) if c.get("feedback")],
    }
    prompt = f"""Write a short narrative (under 300 words, Markdown, no headings
above level 3) explaining this modeling run to a technical reader: what was
tried, why the selected candidate was chosen, and what to try next.
Do NOT quote any numeric scores; the report shows exact numbers in tables.
Do not claim the result meets a target unless acceptance_status is "met".

Facts: {json.dumps(facts, default=str)}"""
    started = time.perf_counter()
    try:
        response = deps.llm.text("report", prompt)
    except Exception as exc:
        return None, [{
            "node": "report", "type": "llm_call", "candidate_id": None, "status": "api_error",
            "llm_attempt": 0, "elapsed_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}", "usage": None,
        }]
    return response.text, [{
        "node": "report", "type": "llm_call", "candidate_id": None, "status": "ok",
        "llm_attempt": 0, "elapsed_s": round(time.perf_counter() - started, 3), "usage": response.usage,
    }]


# --------------------------------------------------------------------------
# Facts derived from state (tolerant of missing fields)
# --------------------------------------------------------------------------

def build_outcome(state, deps) -> dict:
    metric = state.get("primary_metric")
    candidates = state.get("candidates", [])
    best = select_best(candidates, metric) if metric else None
    stop_reason = state.get("stop_reason") or "failed"
    threshold = deps.config.acceptance_threshold
    return {
        "stop_reason": stop_reason,
        "stop_reason_explanation": STOP_REASONS.get(stop_reason, ""),
        "acceptance_status": acceptance_status(best, metric, threshold),
        "acceptance_basis": "validation (CV mean on the development partition)",
        "acceptance_threshold": threshold,
        "selected_candidate_id": best["candidate_id"] if best else None,
        "attempts_used": len(candidates),
        "max_candidates": deps.config.max_candidates,
    }


def usage_summary(events: list[dict], llm_mode: str) -> dict:
    calls = [e for e in events if e.get("type") == "llm_call" and e.get("status") != "api_error"]
    if llm_mode == "offline":
        return {"llm_calls": len(calls), "status": "not applicable (offline mode)"}
    with_usage = [e for e in calls if e.get("usage")]
    summary = {"llm_calls": len(calls), "calls_with_usage": len(with_usage)}
    if not with_usage:
        summary["status"] = "unavailable (the SDK returned no usage metadata)"
        return summary
    summary["input_tokens"] = sum(e["usage"].get("input_tokens") or 0 for e in with_usage)
    summary["output_tokens"] = sum(e["usage"].get("output_tokens") or 0 for e in with_usage)
    summary["status"] = "complete" if len(with_usage) == len(calls) else "partial (some calls lacked usage)"
    in_price = os.environ.get("CLAUDE_INPUT_USD_PER_MTOK")
    out_price = os.environ.get("CLAUDE_OUTPUT_USD_PER_MTOK")
    if in_price and out_price and summary["status"] == "complete":
        try:
            cost = (summary["input_tokens"] * float(in_price) + summary["output_tokens"] * float(out_price)) / 1e6
            summary["estimated_cost_usd"] = round(cost, 4)
            summary["cost_note"] = "estimate from user-supplied per-million-token prices in .env"
        except ValueError:
            summary["cost_note"] = "pricing env vars are not numbers; no estimate made"
    return summary


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _model_label(choice) -> str:
    if not choice:
        return "-"
    params = [f"{k}={choice[k]}" for k in ("n_estimators", "max_depth") if choice.get(k) is not None]
    return choice["model_name"] + (f" ({', '.join(params)})" if params else "")


def _warning_label(warning: str) -> str:
    """'ConvergenceWarning: lbfgs failed...' -> 'convergence warning'."""
    category = warning.split(":", 1)[0]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", category).lower()


def _feature_label(plan) -> str:
    if not plan:
        return "-"
    counts: dict[str, int] = {}
    for step in plan["steps"]:
        counts[step["transform"]] = counts.get(step["transform"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------
# Report content as blocks: ("p", text) ("note", text) ("table", head, rows)
# ("list", items) ("md", text) ("img", path, caption)
# --------------------------------------------------------------------------

def build_sections(state, deps) -> list[tuple[str, list]]:
    cfg, ds = deps.config, deps.dataset
    outcome = build_outcome(state, deps)
    metric = state.get("primary_metric")
    spec = METRICS.get(metric) if metric else None
    candidates = state.get("candidates", [])
    critiques = {c["candidate_id"]: c for c in state.get("critiques", [])}
    split = state.get("split_info", {})
    final_test = state.get("final_test")
    events = state.get("events", [])
    sections = []

    if deps.llm.mode == "offline":
        banner = ("note", "Offline mode: decisions come from the predefined decision script "
                          "(src/demo/offline.py). Splitting, validation, cross-validation, selection, "
                          "and final test scoring run exactly as in live mode.")
    else:
        banner = ("note", f"Live mode: decisions generated by '{deps.llm.model_name}' and "
                          "validated before execution.")

    threshold_text = (
        f"validation {metric} {'>=' if spec.greater_is_better else '<='} {cfg.acceptance_threshold}"
        if cfg.acceptance_threshold is not None and spec
        else "No acceptance target was specified."
    )
    selected = outcome["selected_candidate_id"]
    primary_test = None
    if final_test and "metrics" in final_test and metric:
        primary_test = final_test["metrics"].get(metric)
    best = next((c for c in candidates if c["candidate_id"] == selected), None)
    outcome_rows = [
        ["Stop reason", f"{outcome['stop_reason']}: {outcome['stop_reason_explanation']}"],
        ["Acceptance status (validation)", outcome["acceptance_status"]],
        ["Acceptance target", threshold_text],
        ["Primary metric", f"{metric} ({direction(spec)}, chosen by {state.get('metric_source')})" if spec else "-"],
        ["Selected candidate", selected or "none (no valid candidate)"],
        [f"Selected validation {metric or ''} (CV mean ± std)",
         f"{_fmt(best['validation']['primary']['mean'])} ± {_fmt(best['validation']['primary']['std'])}" if best else "-"],
        [f"Final test {metric or ''} (held out, scored once)", _fmt(primary_test)],
        ["Attempts used", f"{outcome['attempts_used']} of {outcome['max_candidates']}"],
    ]
    if final_test and final_test.get("meets_threshold") is not None:
        outcome_rows.append(["Final test vs. target (informational)",
                             "meets target" if final_test["meets_threshold"] else "below target"])
    outcome_blocks = [banner, ("table", ["Item", "Value"], outcome_rows)]
    sections.append(("Outcome", outcome_blocks))

    # sklearn's roc_auc scorer treats the greater label (classes_[1]) as positive.
    pos = None
    labels = sorted(int(k) for k in split.get("dev_class_counts", {}))
    if len(labels) == 2:
        name = ds.class_names.get(labels[-1])
        pos = f"label {labels[-1]}" + (f" ('{name}')" if name else "")
    usage = usage_summary(events, deps.llm.mode)
    meta_rows = [
        ["Run ID", deps.run_id],
        ["Mode", "offline (decision script)" if deps.llm.mode == "offline" else "live"],
        ["Model identifier", deps.llm.model_name],
        ["Dataset", f"{ds.name} ({ds.task_type}, target '{ds.target_column}')"],
        ["Random seed", cfg.seed],
        ["Partitions", f"development {split.get('n_dev', '-')} rows / final test {split.get('n_test', '-')} rows "
                       f"(test_size={cfg.test_size}, stratified={split.get('stratified', '-')})"],
        ["Validation", f"{cfg.cv_folds}-fold {'stratified ' if ds.task_type == 'classification' else ''}CV on the development partition"],
        ["Max repair attempts per decision", cfg.max_repair_attempts],
        ["LLM usage", ", ".join(f"{k}={v}" for k, v in usage.items())],
    ]
    if pos:
        meta_rows.append(["ROC AUC positive class", pos])
    sections.append(("Run configuration", [("table", ["Item", "Value"], meta_rows)]))

    if candidates:
        valid = [c for c in candidates if c["status"] == "valid"]
        other_metrics = [m for m in (valid[0]["validation"]["metrics"] if valid else {}) if m != metric]
        head = ["Candidate", "Status", "Features", "Model", f"Validation {metric} (mean ± std)"] + \
               [f"Validation {m}" for m in other_metrics] + ["Critic action -> route", "Notes"]
        rows = []
        for c in candidates:
            crit = critiques.get(c["candidate_id"], {})
            route = "-"
            if crit:
                route = f"{crit.get('recommended_action') or 'unavailable'} -> {crit.get('route')}"
                if crit.get("overridden"):
                    route += f" (override: {crit.get('override_reason')})"
            notes = []
            if c["candidate_id"] == selected:
                notes.append("SELECTED")
            if c.get("feature_plan_origin") and c.get("feature_plan_origin") != c["candidate_id"]:
                notes.append(f"features reused from {c['feature_plan_origin']}")
            if c["status"] == "failed":
                f = c["failure"]
                notes.append(f"{f['kind']} at {f['stage']}: {'; '.join(f['messages'])[:200]}")
            # Short label only (e.g. "convergence warning"); full text stays in trace.json.
            for w in sorted({_warning_label(w) for w in c.get("warnings", [])}):
                notes.append(w)
            if c["status"] == "valid":
                p = c["validation"]["primary"]
                scores = [f"{_fmt(p['mean'])} ± {_fmt(p['std'])}"] + \
                         [_fmt(c["validation"]["metrics"][m]["mean"]) for m in other_metrics]
            else:
                scores = ["-"] + ["-" for _ in other_metrics]
            rows.append([c["candidate_id"], c["status"], _feature_label(c.get("feature_plan")),
                         _model_label(c.get("model_choice"))] + scores + [route, "; ".join(notes) or "-"])
        sections.append(("Candidate comparison (validation only)", [
            ("p", "Scores are cross-validation means on the development partition. Selection "
                  "takes the best primary-metric mean; ties go to the lower CV std, then the "
                  "earlier candidate."),
            ("table", head, rows),
        ]))

    if best:
        grouped: dict[str, list[str]] = {}
        for step in best["feature_plan"]["steps"]:
            grouped.setdefault(step["transform"], []).append(step["column"])
        blocks = [("table", ["Transform", "Columns"],
                   [[t, f"({len(cols)}) " + ", ".join(cols)] for t, cols in sorted(grouped.items())]),
                  ("p", f"Model: {_model_label(best['model_choice'])}"),
                  ("p", f"Model reasoning: {best['model_choice']['reasoning']}"),
                  ("p", f"Feature reasoning: {best['feature_plan']['reasoning']}")]
        sections.append((f"Selected candidate: {best['candidate_id']}", blocks))

    if final_test is None:
        ft_blocks = [("p", "Not run: there was no valid candidate to evaluate.")]
    elif "error" in final_test:
        ft_blocks = [("p", f"Final evaluation failed ({final_test['error']['kind']}): "
                           f"{'; '.join(final_test['error']['messages'])}")]
    else:
        ft_blocks = [
            ("p", f"Candidate {final_test['candidate_id']} was refit from scratch on all "
                  f"{final_test['n_train']} development rows and scored once on "
                  f"{final_test['n_test']} held-out rows. These scores were not used for any decision."),
            ("table", ["Metric", "Final test score"], [[m, _fmt(v)] for m, v in final_test["metrics"].items()]),
        ]
    sections.append(("Final test evaluation", ft_blocks))

    if critiques:
        items = []
        for cid, c in critiques.items():
            text = c.get("feedback") or f"critic failed: {c.get('error')}"
            items.append(f"{cid}: {text}")
        sections.append(("Critic feedback", [("list", items)]))

    narrative = state.get("narrative")
    if narrative:
        sections.append(("Narrative", [("md", narrative)]))
    else:
        sections.append(("Narrative", [("p", "No narrative was generated for this run (the call was "
                                             "skipped or failed). The deterministic sections above are complete.")]))

    limits = list(LIMITATIONS)
    if outcome["stop_reason"] == "budget_exhausted":
        limits.insert(0, "The candidate budget ran out before the critic was satisfied; a larger "
                         "budget might find a better candidate.")
    if cfg.acceptance_threshold is None:
        limits.insert(0, "No acceptance target was given, so the run cannot say whether the result "
                         "is good enough for a particular use.")
    sections.append(("Limitations", [("list", limits)]))

    fatal = state.get("fatal_error")
    failed = [c for c in candidates if c["status"] == "failed"]
    if fatal or failed:
        items = []
        if fatal:
            items.append(f"Run-level failure in {fatal['node']} ({fatal['kind']}): {'; '.join(fatal['messages'])}")
        for c in failed:
            f = c["failure"]
            items.append(f"{c['candidate_id']} failed at {f['stage']} ({f['kind']}): {'; '.join(f['messages'])}")
        sections.append(("Failure diagnostics", [("list", items)]))

    figures = state.get("eda_figures") or []
    if figures:
        sections.append(("EDA figures (development partition)", [("img", f, os.path.basename(f)) for f in figures]))

    trace_rows = []
    for e in events:
        detail = {k: v for k, v in e.items()
                  if k not in ("node", "candidate_id", "type", "status", "elapsed_s", "llm_attempt") and v not in (None, [], {})}
        trace_rows.append([e.get("node"), e.get("candidate_id") or "-", e.get("type", "-"),
                           e.get("llm_attempt", "-"), e.get("status"), _fmt(e.get("elapsed_s"), 3),
                           json.dumps(detail, default=str)[:300]])
    sections.append(("Event trace", [
        ("p", "The full machine-readable trace is in trace.json next to this report."),
        ("table", ["Node", "Candidate", "Type", "LLM attempt", "Status", "Seconds", "Details"], trace_rows),
    ]))
    return sections


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(title: str, sections, output_dir: str) -> str:
    lines = [f"# {title}", ""]
    for heading, blocks in sections:
        lines += [f"## {heading}", ""]
        for block in blocks:
            kind = block[0]
            if kind == "p":
                lines += [block[1], ""]
            elif kind == "note":
                lines += [f"> **{block[1]}**", ""]
            elif kind == "md":
                lines += [block[1], ""]
            elif kind == "list":
                lines += [f"- {item}" for item in block[1]] + [""]
            elif kind == "table":
                head, rows = block[1], block[2]
                lines.append("| " + " | ".join(_md_cell(h) for h in head) + " |")
                lines.append("|" + "---|" * len(head))
                lines += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows]
                lines.append("")
            elif kind == "img":
                lines += [f"![{block[2]}]({os.path.relpath(block[1], output_dir)})", ""]
    return "\n".join(lines)


def _md_to_html(md_text: str) -> str:
    """Tiny, escaping Markdown subset for the narrative: headings, bullets, paragraphs."""
    out = []
    for block in md_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if all(l.lstrip().startswith(("- ", "* ")) for l in lines):
            out.append("<ul>" + "".join(f"<li>{html.escape(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        elif block.startswith("### "):
            out.append(f"<h4>{html.escape(block[4:])}</h4>")
        elif block.startswith("## ") or block.startswith("# "):
            out.append(f"<h3>{html.escape(block.lstrip('#').strip())}</h3>")
        else:
            out.append(f"<p>{html.escape(block)}</p>")
    return "\n".join(out)


def render_html(title: str, sections) -> str:
    e = html.escape
    parts = []
    for heading, blocks in sections:
        parts.append(f"<h2>{e(heading)}</h2>")
        for block in blocks:
            kind = block[0]
            if kind == "p":
                parts.append(f"<p>{e(str(block[1]))}</p>")
            elif kind == "note":
                parts.append(f'<p class="note">{e(block[1])}</p>')
            elif kind == "md":
                parts.append(f'<div class="narrative">{_md_to_html(block[1])}</div>')
            elif kind == "list":
                parts.append("<ul>" + "".join(f"<li>{e(str(i))}</li>" for i in block[1]) + "</ul>")
            elif kind == "table":
                head = "".join(f"<th>{e(str(h))}</th>" for h in block[1])
                rows = "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in row) + "</tr>" for row in block[2])
                parts.append(f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
            elif kind == "img" and os.path.exists(block[1]):
                with open(block[1], "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{e(block[2])}">'
                             f"<figcaption>{e(block[2])}</figcaption></figure>")
    body = "\n".join(parts)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1000px; margin: 40px auto;
        padding: 0 16px; line-height: 1.5; color: #1a1a1a; background: #fff; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9em; }}
td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f6f6; }}
.note {{ background: #fff6dd; border-left: 4px solid #e0a800; padding: 8px 12px; }}
figure img {{ max-width: 100%; }}
figcaption {{ font-size: 0.85em; color: #666; }}
</style></head>
<body>
<h1>{e(title)}</h1>
{body}
</body></html>"""


def build_trace(state, deps) -> dict:
    """Machine-readable record of the run. Holds no raw data rows or secrets."""
    plan = state.get("plan")
    return {
        "run_id": deps.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": deps.llm.mode,
        "model_identifier": deps.llm.model_name,
        "dataset": {"name": deps.dataset.name, "task_type": deps.dataset.task_type,
                    "target_column": deps.dataset.target_column, "n_rows": len(deps.dataset.df)},
        "config": deps.config.model_dump(),
        "split": state.get("split_info"),
        "plan": plan.model_dump() if plan else None,
        "primary_metric": state.get("primary_metric"),
        "metric_source": state.get("metric_source"),
        "outcome": build_outcome(state, deps),
        "candidates": state.get("candidates", []),
        "critiques": state.get("critiques", []),
        "final_test": state.get("final_test"),
        "fatal_error": state.get("fatal_error"),
        "llm_usage": usage_summary(state.get("events", []), deps.llm.mode),
        "events": state.get("events", []),
        "notes": "LLM-provided reasoning and feedback are generated explanations, not verified "
                 "analysis or internal reasoning traces.",
    }


def write_reports(state, deps) -> dict:
    output_dir = deps.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mode = " (offline mode)" if deps.llm.mode == "offline" else ""
    title = f"Autonomous Data Scientist run: {deps.dataset.name}{mode}"
    sections = build_sections(state, deps)

    paths = {
        "markdown": os.path.join(output_dir, "report.md"),
        "html": os.path.join(output_dir, "report.html"),
        "trace": os.path.join(output_dir, "trace.json"),
    }
    with open(paths["markdown"], "w") as f:
        f.write(render_markdown(title, sections, output_dir))
    with open(paths["html"], "w") as f:
        f.write(render_html(title, sections))
    with open(paths["trace"], "w") as f:
        json.dump(build_trace(state, deps), f, indent=2, default=str)
    return paths
