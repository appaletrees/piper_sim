import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_geometric_contact_does_not_require_scratch_solver_force(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('chord_geometric_contact', scripts/'probe_chord_geometric_contact.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = {'task/plate': 1, 'right/link7': 2, 'right/link8': 3}
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=ids[name]),
                            geom_bodyid=np.array([1, 2, 3, 4]))
    # No solver buffer or force fields exist in this fixture. Geometric onset
    # must count both fingers even when scratch dynamics would unload one.
    contacts = [SimpleNamespace(geom=(0, 1), dist=-.0001),
                SimpleNamespace(geom=(2, 0), dist=0.)]
    backend = SimpleNamespace(model=model, data=SimpleNamespace(contact=contacts))
    assert module.geometric_fingers(backend, 'right', 'plate') == ['7', '8']
    contacts[1].dist = .0001
    assert module.geometric_fingers(backend, 'right', 'plate') == ['7']
    contacts.append(SimpleNamespace(geom=(2, 3), dist=-.001))
    assert module.geometric_fingers(backend, 'right', 'plate') == ['7']
