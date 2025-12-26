import json
import pandas as pd
from agents.research_agent import ResearchAgent


def test_research_agent_basic(tmp_path):
    # create a small synthetic predictions CSV
    df = pd.DataFrame({
        "y_true": [1.0, 2.0, 3.0, 4.0, 20.0],
        "y_pred": [1.1, 1.9, 3.5, 3.8, 18.0],
        "commodity": ["A", "A", "B", "B", "C"],
    })
    preds = tmp_path / "preds.csv"
    df.to_csv(preds, index=False)

    out = tmp_path / "report.json"
    agent = ResearchAgent(predictions_path=str(preds), out_path=str(out))
    report = agent.run()

    # output file exists
    assert out.exists()
    # report contains expected keys
    assert hasattr(report, "summary_bullets")
    assert isinstance(report.summary_bullets, list)
    assert any("MAE" in b or "Model achieves" in b for b in report.summary_bullets)
    # failure modes should include low sample warnings for commodity C (n=1)
    assert any("Low sample" in f for f in report.failure_modes)

    # test JSON content
    data = json.loads(out.read_text())
    assert "summary_bullets" in data
    assert "key_metrics" in data
