"""LangGraph node functions.

Each node receives the current AgentState and returns a partial state
update (LangGraph merges it in). LLM-driven nodes (plan, feature
engineering, modeling, critic, report) use `with_structured_output` so the
model's decision is a validated pydantic object, not free text to parse.
"""

from .llm import get_llm
from .state import AnalysisPlan, Critique, FeaturePlan, ModelChoice
from . import tools as T


def plan_node(state):
    llm = get_llm().with_structured_output(AnalysisPlan)
    prompt = f"""You are a senior data scientist planning an analysis.

Dataset: {state['dataset_name']}
Task type: {state['task_type']}
Target column: {state['target_column']}
Description: {state['description']}
Columns: {list(state['df'].columns)}

Produce a short analysis plan and name the single primary metric to
optimize for this task type."""
    plan = llm.invoke(prompt)
    trace = state.get("trace", []) + [f"[Planner] {plan.approach_summary}"]
    return {"plan": plan, "trace": trace}


def eda_node(state):
    summary, figures = T.run_eda(state["df"], state["target_column"], "outputs/figures")
    trace = state["trace"] + [
        f"[EDA] {summary['n_rows']} rows, {summary['n_columns']} columns, "
        f"{summary['missing_values']} missing values"
    ]
    return {"eda_summary": summary, "eda_figures": figures, "trace": trace}


def feature_engineering_node(state):
    llm = get_llm().with_structured_output(FeaturePlan)
    feedback = state["critique"].feedback if state.get("critique") else "None yet - first pass."
    prompt = f"""You are engineering features for a {state['task_type']} model.

Target column: {state['target_column']}
Columns and dtypes: {state['df'].dtypes.astype(str).to_dict()}
EDA summary: {state['eda_summary']}
Prior critique feedback: {feedback}

For every non-target column, decide one transform: scale, log1p, drop, or
keep. Return your reasoning and the full list of steps."""
    feature_plan = llm.invoke(prompt)
    engineered_df = T.apply_feature_plan(state["df"], state["target_column"], feature_plan)
    trace = state["trace"] + [f"[FeatureEngineering] {feature_plan.reasoning}"]
    return {"feature_plan": feature_plan, "engineered_df": engineered_df, "trace": trace}


def modeling_node(state):
    llm = get_llm().with_structured_output(ModelChoice)
    valid_models = (
        ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
        if state["task_type"] == "classification"
        else ["linear_regression", "random_forest_regressor", "gradient_boosting_regressor"]
    )
    feedback = state["critique"].feedback if state.get("critique") else "None yet - first pass."
    prompt = f"""Choose a scikit-learn model for this {state['task_type']} task.

Valid model_name options: {valid_models}
Primary metric to optimize: {state['plan'].primary_metric}
Prior critique feedback: {feedback}

Pick model_name, plus n_estimators/max_depth if the model uses them."""
    model_choice = llm.invoke(prompt)
    metrics = T.train_and_evaluate(
        state["engineered_df"], state["target_column"], state["task_type"], model_choice
    )
    trace = state["trace"] + [f"[Modeling] {model_choice.model_name} -> {metrics}"]
    return {"model_choice": model_choice, "metrics": metrics, "trace": trace}


def critic_node(state):
    llm = get_llm().with_structured_output(Critique)
    iteration = state.get("iteration", 0) + 1
    prompt = f"""You are reviewing a modeling result.

Primary metric: {state['plan'].primary_metric}
Metrics achieved: {state['metrics']}
Iteration: {iteration} of {state['max_iterations']}

Decide if this is good enough to ship (satisfied=true,
recommended_action='proceed'), or whether feature engineering
('iterate_features') or model choice ('iterate_model') should be
revisited. If iteration >= max_iterations you must choose 'proceed'
regardless of quality."""
    critique = llm.invoke(prompt)
    if iteration >= state["max_iterations"]:
        critique.recommended_action = "proceed"
    trace = state["trace"] + [
        f"[Critic] satisfied={critique.satisfied} action={critique.recommended_action}: "
        f"{critique.feedback}"
    ]
    return {"critique": critique, "iteration": iteration, "trace": trace}


def route_after_critic(state):
    return state["critique"].recommended_action


def report_node(state):
    from ..report.generator import generate_report

    path = generate_report(state)
    trace = state["trace"] + [f"[Report] written to {path}"]
    return {"report_path": path, "trace": trace}
