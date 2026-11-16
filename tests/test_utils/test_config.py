import pytest
from reconciliacao.utils.config import load_config


def test_given_a_config_file_when_loaded_then_a_dict_comes_back(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("simulation:\n  n_records: 100\n  random_seed: 42\n")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    assert cfg["simulation"]["n_records"] == 100


def test_given_a_missing_config_file_when_loaded_then_it_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_given_no_path_when_loaded_then_the_project_config_is_read():
    cfg = load_config("config.yaml")
    assert cfg["simulation"]["n_records"] == 7500
    assert "exact" in cfg["scenarios"]
    assert "fuzzy" in cfg["scenarios"]
