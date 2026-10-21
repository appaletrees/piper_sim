from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path=None):
    path = Path(path or ROOT / "configs/workcell.yaml").resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg
