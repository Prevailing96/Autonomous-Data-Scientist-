"""Entry point: python cli.py --dataset breast_cancer"""

import argparse
import os

from dotenv import load_dotenv

from src.agent.graph import build_graph
from src.data.datasets import AVAILABLE_DATASETS, load_dataset


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Autonomous Data Scientist - a LangGraph multi-agent pipeline"
    )
    parser.add_argument("--dataset", default="breast_cancer", choices=AVAILABLE_DATASETS)
    parser.add_argument("--max-iterations", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (see .env.example) before running.")

    bundle = load_dataset(args.dataset)
    graph = build_graph()

    initial_state = {
        "dataset_name": bundle.name,
        "task_type": bundle.task_type,
        "target_column": bundle.target_column,
        "description": bundle.description,
        "df": bundle.df,
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "trace": [],
    }

    print(f"Running autonomous data science agent on '{bundle.name}'...\n")
    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    print("\n--- Agent trace ---")
    for line in final_state["trace"]:
        print(line)
    print(f"\nDone. Report written to: {final_state['report_path']}")


if __name__ == "__main__":
    main()
