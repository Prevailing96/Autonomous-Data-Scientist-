"""Entry point.

    python cli.py --dataset breast_cancer --offline          # predefined decisions, no API key
    python cli.py --dataset breast_cancer                    # live Claude run
"""

import argparse
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from src.agent.errors import ConfigError, DatasetError
from src.agent.llm import ClaudeLLM, resolve_model_name
from src.agent.metrics import METRICS
from src.agent.runner import run_pipeline
from src.agent.state import RunConfig
from src.data.datasets import AVAILABLE_DATASETS, load_dataset


def main():
    load_dotenv()  # before any model configuration is read
    parser = argparse.ArgumentParser(description="Autonomous Data Scientist: a bounded LangGraph workflow")
    parser.add_argument("--dataset", default="breast_cancer", choices=AVAILABLE_DATASETS)
    parser.add_argument("--offline", action="store_true",
                        help="use the predefined decision script (src/demo/offline.py)")
    parser.add_argument("--metric", choices=sorted(METRICS), help="primary metric (overrides the planner)")
    parser.add_argument("--threshold", type=float, help="acceptance threshold on validation score (needs --metric)")
    parser.add_argument("--max-candidates", type=int, default=3, help="max candidate attempts, failures included")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    try:
        config = RunConfig(
            dataset=args.dataset,
            primary_metric=args.metric,
            acceptance_threshold=args.threshold,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration:\n{exc}")

    bundle = load_dataset(args.dataset)
    if args.offline:
        from src.demo.offline import build_offline_llm

        llm = build_offline_llm(bundle)
        print("Offline mode: using the predefined decision script (src/demo/offline.py).")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (see .env.example), or use --offline.")
        try:
            llm = ClaudeLLM(resolve_model_name())
        except ConfigError as exc:
            raise SystemExit(str(exc))
        print(f"Live mode: model '{llm.model_name}'.")

    try:
        state = run_pipeline(bundle, config, llm, output_root=args.output_dir)
    except (ConfigError, DatasetError) as exc:
        raise SystemExit(f"Cannot run: {exc}")

    outcome_line = f"stop_reason={state.get('stop_reason')}  selected={state.get('best_candidate_id')}"
    print(f"\nRun {state['run_id']} finished in {state['total_elapsed_s']}s: {outcome_line}")
    for c in state.get("candidates", []):
        if c["status"] == "valid":
            p = c["validation"]["primary"]
            print(f"  {c['candidate_id']}: {c['model_choice']['model_name']:<30} "
                  f"validation {state['primary_metric']}={p['mean']:.4f} ± {p['std']:.4f}")
        else:
            print(f"  {c['candidate_id']}: FAILED ({c['failure']['kind']} at {c['failure']['stage']})")
    final = state.get("final_test")
    if final and "metrics" in final:
        print(f"  final test {state['primary_metric']} ({final['candidate_id']}): "
              f"{final['metrics'][state['primary_metric']]:.4f}")
    for kind, path in state.get("report_paths", {}).items():
        print(f"  {kind:<8} {path}")


if __name__ == "__main__":
    main()
