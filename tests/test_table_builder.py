import os
import tempfile
from bayesprompt.evaluation.build_tables import run_build_tables

def test_build_tables_with_mock_data():
    metrics_dir = "tests/fixtures/metrics"
    with tempfile.TemporaryDirectory() as out_dir:
        run_build_tables(metrics_dir, out_dir, template=False)
        assert os.path.exists(os.path.join(out_dir, "table1_crossmodality.tex"))
        assert os.path.exists(os.path.join(out_dir, "table2_calibration_stability.tex"))
        assert os.path.exists(os.path.join(out_dir, "table3_ablation.tex"))

def test_build_tables_template():
    with tempfile.TemporaryDirectory() as out_dir:
        run_build_tables("nonexistent_dir", out_dir, template=True)
        assert os.path.exists(os.path.join(out_dir, "table1_crossmodality.tex"))
        
        with open(os.path.join(out_dir, "table1_crossmodality.tex"), 'r') as f:
            content = f.read()
            assert "Data not available" in content
