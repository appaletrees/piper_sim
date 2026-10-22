"""Task registry: add task implementations/adapters without editing robot backends."""
import importlib
from pathlib import Path
import yaml
from ..config import ROOT

# Entries are trusted project code, never arbitrary import paths from task YAML.
REGISTRY = {
    'charger_socket': ('piper_sim.tasks.connector', 'ConnectorTask', {
        'mujoco': 'piper_sim.tasks.mujoco', 'isaac': 'piper_sim.tasks.isaac'}),
}


def create_task(config, workcell):
    if isinstance(config, (str, Path)):
        path=Path(config)
        if not path.is_absolute(): path=ROOT/path
        config=yaml.safe_load(path.read_text())
    name=config.get('name')
    if name not in REGISTRY:
        raise ValueError(f'Unknown task {name!r}; registered: {list(REGISTRY)}')
    module, cls, _=REGISTRY[name]
    return getattr(importlib.import_module(module),cls)(config,workcell)


def engine_adapter(task, engine):
    return importlib.import_module(REGISTRY[task.config['name']][2][engine])
