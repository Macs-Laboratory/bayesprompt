import os
import tempfile
from bayesprompt.evaluation.check_baselines import run_check_baselines

def test_check_baselines():
    metrics_dir = "tests/fixtures/metrics"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "coverage.md")
        run_check_baselines(metrics_dir, out_path)
        assert os.path.exists(out_path)
        
        with open(out_path, 'r') as f:
            content = f.read()
            assert "U-Net" in content
            assert "BayesPrompt" in content
