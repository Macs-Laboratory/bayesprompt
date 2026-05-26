import os
import json
import tempfile
import pytest
import subprocess

def test_split_validation_clean():
    clean_split = {
        "train": [{"patient_id": "P01", "image": "img1"}],
        "val": [{"patient_id": "P02", "image": "img2"}],
        "test": [{"patient_id": "P03", "image": "img3"}]
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(clean_split, f)
        temp_path = f.name
        
    result = subprocess.run(["python", "-m", "bayesprompt.datasets.validate_splits", temp_path], capture_output=True)
    assert result.returncode == 0
    os.remove(temp_path)

def test_split_validation_leakage():
    leak_split = {
        "train": [{"patient_id": "P01", "image": "img1"}],
        "val": [{"patient_id": "P02", "image": "img2"}],
        "test": [{"patient_id": "P01", "image": "img3"}]
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(leak_split, f)
        temp_path = f.name
        
    result = subprocess.run(["python", "-m", "bayesprompt.datasets.validate_splits", temp_path], capture_output=True)
    assert result.returncode != 0
    os.remove(temp_path)
