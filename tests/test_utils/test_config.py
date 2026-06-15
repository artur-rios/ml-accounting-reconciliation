import pytest
from pathlib import Path
from reconciliacao.utils.config import load_config


def test_load_config_returns_dict(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("simulation:\n  n_records: 100\n  random_seed: 42\n")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    assert cfg["simulation"]["n_records"] == 100


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_default_path_reads_project_config():
    cfg = load_config("config.yaml")
    assert cfg["simulation"]["n_records"] == 7500
    assert "exact" in cfg["scenarios"]
    assert "fuzzy" in cfg["scenarios"]
