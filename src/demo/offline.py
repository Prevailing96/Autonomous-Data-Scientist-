"""Predefined decision script for offline mode (`python cli.py --offline`).

Supplies the plan, feature, model, and critic decisions in place of live
model calls, so the full workflow (validation and repair, a feature
refinement, a model refinement, selection, final test, reporting) runs
without an API key. Everything downstream of the decisions runs unchanged.

Script, per dataset:
  plan     -> recommends a task-appropriate metric
  C1 feat  -> first response omits most columns (semantic error), then a
              repaired plan that keeps every feature
  C1 model -> linear baseline;           critic: iterate_features
  C2 feat  -> scale every feature
  C2 model -> same linear baseline;      critic: iterate_model
  C3 model -> gradient boosting (reuses C2 features); critic: proceed
"""

from ..agent.fake_llm import ScriptedLLM

NARRATIVE = (
    "The run started from a linear baseline on unscaled features, standardized the features "
    "on the critic's advice, then tried a gradient-boosted model on the same preprocessing. "
    "The selected candidate is whichever had the best cross-validated primary metric; see "
    "the tables for exact scores.\n\n"
    "- Next step: try a different seed to see how stable the ranking is."
)


def build_offline_llm(bundle) -> ScriptedLLM:
    features = [c for c in bundle.df.columns if c != bundle.target_column]
    if bundle.task_type == "regression":
        metric, linear, boosted = "rmse", "linear_regression", "gradient_boosting_regressor"
    else:
        binary = bundle.df[bundle.target_column].nunique() == 2
        metric = "roc_auc" if binary else "f1_weighted"
        linear, boosted = "logistic_regression", "gradient_boosting_classifier"

    def plan(transform, reasoning):
        return {"reasoning": reasoning, "steps": [{"column": c, "transform": transform} for c in features]}

    script = {
        "plan": [{
            "approach_summary": f"Start from a linear baseline, then test whether scaling "
                                f"and a tree ensemble improve {metric}.",
            "recommended_metric": metric,
        }],
        "feature_engineering": [
            # Deliberately incomplete: semantic validation rejects it, one repair follows.
            {"reasoning": "Start with the first feature.", "steps": [{"column": features[0], "transform": "keep"}]},
            plan("keep", "Baseline: use every feature as-is."),
            plan("scale", "Standardize features so the linear model is not dominated by large-scale columns."),
        ],
        "modeling": [
            {"reasoning": "Linear baseline.", "model_name": linear},
            {"reasoning": "Same model, now on scaled features.", "model_name": linear},
            {"reasoning": "Try a small boosted ensemble.", "model_name": boosted,
             "n_estimators": 100, "max_depth": 2},
        ],
        "critic": [
            {"recommended_action": "iterate_features",
             "feedback": "Features are on very different scales; standardize them."},
            {"recommended_action": "iterate_model",
             "feedback": "Preprocessing looks reasonable; try a non-linear model."},
            {"recommended_action": "proceed",
             "feedback": "Further changes are unlikely to help much; report the best candidate."},
        ],
    }
    return ScriptedLLM(script, narrative=NARRATIVE)
