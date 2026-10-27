"""Runs one isolated pipeline execution: preflight, split, graph, reports.

The development/test split happens here, before the graph starts. Graph
state only ever carries the development partition; the test partition is
bound to the final_evaluation node and nothing else.
"""

import os
import time
import uuid
from datetime import datetime, timezone

from . import tools as T
from .errors import ConfigError
from .graph import build_graph, recursion_limit
from .nodes import Deps
from .state import RunConfig
from .validation import validate_metric


def new_run_id(dataset_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{dataset_name}-{uuid.uuid4().hex[:6]}"


def run_pipeline(bundle, config: RunConfig, llm, output_root: str = "outputs") -> dict:
    """Execute a run and return its final state (including `report_paths`)."""
    # Preflight: fail clearly before any LLM call or file is written.
    T.validate_dataset(bundle.df, bundle.target_column, bundle.task_type)
    n_classes = int(bundle.df[bundle.target_column].nunique()) if bundle.task_type == "classification" else None
    if config.primary_metric:
        problems = validate_metric(config.primary_metric, bundle.task_type, n_classes)
        if problems:
            raise ConfigError(problems[0])

    dev_df, test_df, split_info = T.split_dataset(
        bundle.df, bundle.target_column, bundle.task_type, config.test_size, config.seed
    )
    T.make_cv(dev_df[bundle.target_column], bundle.task_type, config.cv_folds, config.seed)

    run_id = new_run_id(bundle.name)
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=False)  # never reuse another run's directory

    deps = Deps(llm=llm, config=config, dataset=bundle, run_id=run_id, output_dir=output_dir)
    graph = build_graph(deps, holdout=test_df)
    state = {
        "dev_df": dev_df,
        "split_info": split_info,
        "candidates": [],
        "critiques": [],
        "events": [{"node": "runner", "type": "node", "status": "ok", "candidate_id": None,
                    "elapsed_s": 0.0, "detail": "split dataset", **{k: split_info[k] for k in ("n_dev", "n_test", "stratified")}}],
    }

    started = time.perf_counter()
    try:
        for snapshot in graph.stream(
            state, config={"recursion_limit": recursion_limit(config.max_candidates)}, stream_mode="values"
        ):
            state = snapshot
    except Exception as exc:
        # Something escaped the nodes' own error handling. Keep what we have
        # and still write an honest failure report.
        from ..report.generator import write_reports

        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        state = {
            **state,
            "fatal_error": {"node": "graph", "kind": "internal_error", "messages": [message]},
            "stop_reason": "failed",
            "events": state.get("events", []) + [{"node": "runner", "type": "node", "candidate_id": None,
                                                  "status": "internal_error", "error": message,
                                                  "elapsed_s": round(time.perf_counter() - started, 3)}],
        }
        state["report_paths"] = write_reports(state, deps)
    state["run_id"] = run_id
    state["total_elapsed_s"] = round(time.perf_counter() - started, 3)
    return state
