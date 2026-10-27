"""End-to-end behavior of the graph with scripted LLM decisions."""

import json

from src.agent import nodes as N
from src.agent import tools as T
from src.data.datasets import load_dataset

from .conftest import calls_to, critic, feature_plan, model, plan, run

BC = load_dataset("breast_cancer")


def _cand(state, cid):
    return next(c for c in state["candidates"] if c["candidate_id"] == cid)


def test_critic_feature_refinement_builds_new_feature_plan(tmp_path):
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC, "keep"), feature_plan(BC, "scale")],
        "modeling": [model(), model()],
        "critic": [critic("iterate_features"), critic("proceed")],
    })
    assert len(calls_to(llm, "feature_engineering")) == 2
    assert _cand(state, "C2")["feature_plan_origin"] == "C2"
    assert {s["transform"] for s in _cand(state, "C2")["feature_plan"]["steps"]} == {"scale"}
    assert state["stop_reason"] == "critic_proceeded"
    # The critic's feedback reaches the next feature-engineering prompt.
    assert "test feedback" in calls_to(llm, "feature_engineering")[1]


def test_critic_model_refinement_reuses_feature_plan(tmp_path):
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)],
        "modeling": [model(), model("random_forest_classifier", n_estimators=20, max_depth=3)],
        "critic": [critic("iterate_model"), critic("proceed")],
    })
    assert len(calls_to(llm, "feature_engineering")) == 1
    c2 = _cand(state, "C2")
    assert c2["feature_plan_origin"] == "C1"
    assert c2["model_choice"]["model_name"] == "random_forest_classifier"


def test_critic_proceeds_immediately(tmp_path):
    state, llm = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    })
    assert [c["candidate_id"] for c in state["candidates"]] == ["C1"]
    assert state["stop_reason"] == "critic_proceeded"
    assert state["final_test"]["candidate_id"] == "C1"


def test_code_enforces_budget_when_critic_keeps_iterating(tmp_path):
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)] * 5,
        "modeling": [model()] * 5,
        "critic": [critic("iterate_features")] * 5,
    }, max_candidates=2)
    assert len(state["candidates"]) == 2
    assert len(calls_to(llm, "critic")) == 2
    assert state["stop_reason"] == "budget_exhausted"
    last = state["critiques"][-1]
    assert last["recommended_action"] == "iterate_features"  # critic's wish is preserved
    assert last["overridden"] and "budget" in last["override_reason"]
    assert state["final_test"] is not None


def test_worse_later_candidate_does_not_replace_best(tmp_path):
    keep_one = {c: "drop" for c in list(BC.df.columns)[1:-1]}  # only the first feature survives
    state, _ = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC), feature_plan(BC, "scale", keep_one)],
        "modeling": [model(), model()],
        "critic": [critic("iterate_features"), critic("proceed")],
    })
    c1, c2 = _cand(state, "C1"), _cand(state, "C2")
    assert c2["validation"]["primary"]["mean"] < c1["validation"]["primary"]["mean"]
    assert state["best_candidate_id"] == "C1"
    assert state["final_test"]["candidate_id"] == "C1"


def test_rmse_is_minimized_end_to_end(tmp_path):
    db = load_dataset("diabetes")
    state, _ = run(tmp_path, {
        "plan": [plan("rmse")],
        "feature_engineering": [feature_plan(db)],
        "modeling": [model("gradient_boosting_regressor", n_estimators=10, max_depth=1),
                     model("linear_regression")],
        "critic": [critic("iterate_model"), critic("proceed")],
    }, dataset="diabetes")
    c1, c2 = _cand(state, "C1"), _cand(state, "C2")
    assert c1["validation"]["primary"]["mean"] > 0  # positive RMSE, not sklearn's negated score
    assert c2["validation"]["primary"]["mean"] < c1["validation"]["primary"]["mean"]
    assert state["best_candidate_id"] == "C2"
    assert state["final_test"]["metrics"]["rmse"] > 0


def test_no_valid_candidates_produces_failure_report(tmp_path):
    bad_model = model("linear_regression")  # regression model on a classification task
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)] * 2,
        "modeling": [bad_model] * 4,
        "critic": [critic("proceed"), critic("proceed")],
    }, max_candidates=2)
    assert all(c["status"] == "failed" for c in state["candidates"])
    assert len(state["candidates"]) == 2
    # Critic said proceed with nothing valid: code overrode it to try again.
    assert state["critiques"][0]["overridden"]
    assert state["stop_reason"] == "failed"
    assert state["final_test"] is None
    report = open(state["report_paths"]["markdown"]).read()
    assert "Not run: there was no valid candidate" in report
    assert "semantic_error at modeling" in report


def test_repair_attempts_stop_at_limit(tmp_path):
    missing_columns = {"reasoning": "x", "steps": [{"column": "mean radius", "transform": "keep"}]}
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [missing_columns] * 10,
        "critic": [critic("iterate_features")] * 10,
    }, max_candidates=2)
    # One original call + one repair per attempt, two attempts, never more.
    assert len(calls_to(llm, "feature_engineering")) == 4
    assert "Your previous response was rejected" in calls_to(llm, "feature_engineering")[1]
    repairs = [e for e in state["events"] if e.get("is_repair")]
    assert len(repairs) == 2
    assert all(c["failure"]["kind"] == "semantic_error" for c in state["candidates"])


def test_repair_disabled_makes_one_call(tmp_path):
    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [{"reasoning": "x", "steps": []}],
        "critic": [critic("proceed")],
    }, max_candidates=1, max_repair_attempts=0)
    assert len(calls_to(llm, "feature_engineering")) == 1
    assert state["stop_reason"] == "failed"


def test_schema_failure_is_classified(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)],
        "modeling": [{"reasoning": "x", "model_name": "xgboost"}] * 2,  # not in the schema
        "critic": [critic("proceed")],
    }, max_candidates=1)
    assert state["candidates"][0]["failure"]["kind"] == "schema_error"


def test_api_failure_preserves_diagnostics(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)],
        "modeling": [ConnectionError("upstream overloaded (529)")],
        "critic": [critic("proceed")],
    }, max_candidates=1)
    failure = state["candidates"][0]["failure"]
    assert failure["kind"] == "api_error"
    assert "upstream overloaded (529)" in failure["messages"][0]
    trace = json.load(open(state["report_paths"]["trace"]))
    assert any(e["status"] == "api_error" and "529" in e.get("error", "") for e in trace["events"])


def test_planner_api_failure_still_writes_report(tmp_path):
    state, _ = run(tmp_path, {"plan": [TimeoutError("request timed out")]})
    assert state["fatal_error"]["kind"] == "api_error"
    assert state["stop_reason"] == "failed"
    report = open(state["report_paths"]["markdown"]).read()
    assert "request timed out" in report
    assert "Not run" in report


def test_tool_failure_preserves_diagnostics(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("fold 3 exploded")

    monkeypatch.setattr(N.T, "evaluate_candidate", boom)
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    }, max_candidates=1)
    failure = state["candidates"][0]["failure"]
    assert failure["kind"] == "tool_error" and "fold 3 exploded" in failure["messages"][0]
    assert state["candidates"][0]["model_choice"]["model_name"] == "logistic_regression"


def test_test_partition_never_reaches_iterative_steps(tmp_path, monkeypatch):
    seen_rows, order = [], []
    real_eda, real_eval, real_final = T.run_eda, T.evaluate_candidate, T.final_evaluate

    def spy_eda(df, *a, **k):
        seen_rows.append(set(df.index)); order.append("eda")
        return real_eda(df, *a, **k)

    def spy_eval(dev_df, *a, **k):
        seen_rows.append(set(dev_df.index)); order.append("cv")
        return real_eval(dev_df, *a, **k)

    def spy_final(*a, **k):
        order.append("final")
        return real_final(*a, **k)

    monkeypatch.setattr(N.T, "run_eda", spy_eda)
    monkeypatch.setattr(N.T, "evaluate_candidate", spy_eval)
    monkeypatch.setattr(N.T, "final_evaluate", spy_final)

    state, llm = run(tmp_path, {
        "plan": [plan()],
        "feature_engineering": [feature_plan(BC)] * 2,
        "modeling": [model()] * 2,
        "critic": [critic("iterate_features"), critic("proceed")],
    })
    _, test_df, _ = T.split_dataset(BC.df, BC.target_column, BC.task_type, 0.2, 42)
    test_rows = set(test_df.index)
    assert seen_rows and all(rows.isdisjoint(test_rows) for rows in seen_rows)
    # Final evaluation happens exactly once, after every CV run.
    assert order == ["eda", "cv", "cv", "final"]
    # Nothing in the graph state handed to nodes carries the test partition.
    assert len(state["dev_df"]) == len(BC.df) - len(test_df)
    # No prompt to the critic (or any other node) mentions final-test results.
    assert all("final test" not in p.lower() for _, p in llm.calls if _ != "report")


def test_user_metric_overrides_planner_and_drives_scoring(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan("roc_auc")], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    }, primary_metric="accuracy")
    assert state["primary_metric"] == "accuracy" and state["metric_source"] == "user"
    c1 = state["candidates"][0]
    assert c1["validation"]["primary_metric"] == "accuracy"
    assert c1["validation"]["primary"] == c1["validation"]["metrics"]["accuracy"]


def test_threshold_met_stops_even_if_critic_wants_more(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)] * 3,
        "modeling": [model()] * 3, "critic": [critic("iterate_features")] * 3,
    }, primary_metric="roc_auc", acceptance_threshold=0.9)
    assert len(state["candidates"]) == 1
    assert state["stop_reason"] == "target_met"
    assert state["critiques"][0]["overridden"]
    trace = json.load(open(state["report_paths"]["trace"]))
    assert trace["outcome"]["acceptance_status"] == "met"


def test_unreachable_threshold_reports_not_met(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    }, primary_metric="roc_auc", acceptance_threshold=1.0)
    trace = json.load(open(state["report_paths"]["trace"]))
    assert trace["outcome"]["stop_reason"] == "critic_proceeded"
    assert trace["outcome"]["acceptance_status"] == "not_met"
    report = open(state["report_paths"]["markdown"]).read()
    assert "| Acceptance status (validation) | not_met |" in report


def test_report_separates_validation_and_test_and_escapes_html(tmp_path):
    evil = "<script>alert(1)</script>"
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed", feedback=evil)],
    }, narrative=f"Narrative with {evil}")
    md = open(state["report_paths"]["markdown"]).read()
    page = open(state["report_paths"]["html"]).read()
    assert "Selected validation roc_auc (CV mean ± std)" in md
    assert "Final test roc_auc (held out, scored once)" in md
    assert "| Stop reason | critic_proceeded" in md
    assert "| Acceptance status (validation) | not_specified |" in md
    assert evil not in page and "&lt;script&gt;" in page
    trace = json.load(open(state["report_paths"]["trace"]))
    assert trace["final_test"]["metrics"]["roc_auc"] == state["final_test"]["metrics"]["roc_auc"]
    assert trace["candidates"][0]["validation"]["primary"]["mean"] > 0


def test_narrative_failure_still_writes_deterministic_report(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    }, narrative=RuntimeError("narrative call failed"))
    md = open(state["report_paths"]["markdown"]).read()
    assert "No narrative was generated" in md
    assert "Final test roc_auc" in md


def test_runs_get_separate_output_directories(tmp_path):
    script = lambda: {"plan": [plan()], "feature_engineering": [feature_plan(BC)],
                      "modeling": [model()], "critic": [critic("proceed")]}
    s1, _ = run(tmp_path, script())
    s2, _ = run(tmp_path, script())
    assert s1["run_id"] != s2["run_id"]
    assert s1["report_paths"]["html"] != s2["report_paths"]["html"]
    assert len(list(tmp_path.iterdir())) == 2


def test_trace_records_run_metadata_without_data_rows(tmp_path):
    state, _ = run(tmp_path, {
        "plan": [plan()], "feature_engineering": [feature_plan(BC)],
        "modeling": [model()], "critic": [critic("proceed")],
    }, seed=7)
    trace = json.load(open(state["report_paths"]["trace"]))
    assert trace["config"]["seed"] == 7 and trace["split"]["seed"] == 7
    assert trace["dataset"]["name"] == "breast_cancer"
    assert trace["model_identifier"] == "offline decision script"
    assert trace["llm_usage"]["status"].startswith("not applicable")
    assert "dev_df" not in json.dumps(trace)
    events = trace["events"]
    assert all({"node", "status", "elapsed_s", "candidate_id"} <= set(e) for e in events)
    critic_event = next(e for e in events if e["node"] == "critic" and e["type"] == "node")
    assert critic_event["route"] == "stop" and critic_event["stop_reason"] == "critic_proceeded"


def test_offline_demo_exercises_repair_and_both_refinements(tmp_path):
    from src.agent.runner import run_pipeline
    from src.agent.state import RunConfig
    from src.demo.offline import build_offline_llm

    state = run_pipeline(BC, RunConfig(dataset="breast_cancer"), build_offline_llm(BC), str(tmp_path))
    actions = [c["recommended_action"] for c in state["critiques"]]
    assert actions == ["iterate_features", "iterate_model", "proceed"]
    assert any(e.get("is_repair") for e in state["events"])
    assert state["final_test"] is not None
    md = open(state["report_paths"]["markdown"]).read()
    assert "(offline mode)" in md and "predefined decision script" in md
