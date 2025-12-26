"""Small CLI wrapper to run the research agent with example flags."""
from agents.research_agent import ResearchAgent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ResearchAgent on model artifacts and predictions")
    parser.add_argument("--metrics", help="Path to metrics JSON", default=None)
    parser.add_argument("--predictions", help="Path to predictions CSV (y_true,y_pred)", default=None)
    parser.add_argument("--out", help="Output JSON path", default="research_report.json")
    args = parser.parse_args()

    agent = ResearchAgent(metrics_path=args.metrics, predictions_path=args.predictions, out_path=args.out)
    report = agent.run()
    print("Wrote:", args.out)
