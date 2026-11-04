from pathlib import Path

import yaml

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def test_train_only_matches_pipeline_train_step():
    pipeline = list(yaml.safe_load_all((DEPLOY / "train.sky.yaml").read_text()))
    train_only = yaml.safe_load((DEPLOY / "train_only.sky.yaml").read_text())
    assert pipeline[-1] == train_only
