"""Turns a finished AgentState into a Markdown report and a styled,
self-contained HTML report (images inlined as base64, no external assets)."""

import base64
import os
from datetime import datetime, timezone

from ..agent.llm import get_llm


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _narrative(state) -> str:
    llm = get_llm()
    prompt = f"""Write a concise, well-organized narrative report in Markdown
for a data science project. Include sections: Executive Summary, Dataset,
Methodology, Results, and Recommendations. Do not include image tags - just
narrative text, under 500 words. Use these facts:

Dataset: {state['dataset_name']} ({state['description']})
Task type: {state['task_type']}
Plan: {state['plan'].approach_summary} (primary metric: {state['plan'].primary_metric})
Feature engineering reasoning: {state['feature_plan'].reasoning}
Model: {state['model_choice'].model_name} (reasoning: {state['model_choice'].reasoning})
Final metrics: {state['metrics']}
Critic verdict: satisfied={state['critique'].satisfied}, feedback={state['critique'].feedback}
Iterations run: {state['iteration']}"""
    return llm.invoke(prompt).content


def _markdown_to_html_blocks(md_text: str) -> str:
    blocks = []
    for block in md_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("### "):
            blocks.append(f"<h4>{block[4:]}</h4>")
        elif block.startswith("## "):
            blocks.append(f"<h3>{block[3:]}</h3>")
        elif block.startswith("# "):
            blocks.append(f"<h2>{block[2:]}</h2>")
        else:
            blocks.append(f"<p>{block}</p>")
    return "\n".join(blocks)


def generate_report(state, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    narrative = _narrative(state)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md_lines = [
        f"# Autonomous Data Science Report: {state['dataset_name']}",
        f"_Generated {timestamp}_",
        "",
        narrative,
        "",
        "## Final Metrics",
        "",
    ]
    for k, v in state["metrics"].items():
        md_lines.append(f"- **{k}**: {v:.4f}")
    md_lines += ["", "## Agent Trace", ""]
    for line in state["trace"]:
        md_lines.append(f"- {line}")
    md_lines += ["", "## Figures", ""]
    for fig in state["eda_figures"]:
        md_lines.append(f"![{os.path.basename(fig)}]({os.path.relpath(fig, output_dir)})")

    md_path = os.path.join(output_dir, f"{state['dataset_name']}_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    fig_html = "".join(
        f'<figure><img src="data:image/png;base64,{_b64_image(fig)}" '
        f'alt="{os.path.basename(fig)}"><figcaption>{os.path.basename(fig)}'
        f"</figcaption></figure>"
        for fig in state["eda_figures"]
    )
    trace_html = "".join(f"<li>{line}</li>" for line in state["trace"])
    metrics_html = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in state["metrics"].items()
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{state['dataset_name']} report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 860px;
        margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #1a1a1a; }}
h1, h2, h3 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
figure {{ margin: 20px 0; }}
figure img {{ max-width: 100%; }}
figcaption {{ font-size: 0.85em; color: #666; }}
code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
</style></head>
<body>
<h1>Autonomous Data Science Report: {state['dataset_name']}</h1>
<p><em>Generated {timestamp}</em></p>
<div>{_markdown_to_html_blocks(narrative)}</div>
<h2>Final Metrics</h2>
<table>{metrics_html}</table>
<h2>Figures</h2>
{fig_html}
<h2>Agent Trace</h2>
<ul>{trace_html}</ul>
</body></html>"""

    html_path = os.path.join(output_dir, f"{state['dataset_name']}_report.html")
    with open(html_path, "w") as f:
        f.write(html)

    return html_path
