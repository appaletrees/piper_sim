"""Check the remaining 32 grid candidates without repeating the first 256.

Verifies the complete ranking prefix against the previous checked order. This
extends the isolated geometric diagnostic, never a live rollout or gate bypass.
"""
import json
from pathlib import Path
import numpy as np
from piper_sim.collection_jobs import sha256
import probe_flat_plate as base
import probe_flat_plate_routes as routes

prior_path = Path(__file__).with_name('flat_routes.json')
prior = json.loads(prior_path.read_text())
original_rank = base.diverse_candidates
original_search = routes.search_with_routes


def remaining(candidates, nominal, tcp, limit):
    ranked = original_rank(candidates, nominal, tcp, len(candidates))
    checked = prior['attempts']
    for expected, actual in zip(checked, ranked):
        assert expected['yaw_deg'] == actual[6]
        np.testing.assert_array_equal(expected['slot_xy_m'], actual[1])
    assert len(ranked) == prior['geometric_candidates'] and len(checked) == 256
    return ranked[len(checked):]


def search(teacher, diagnostics):
    diagnostics['continuation'] = {'script_sha256': sha256(__file__),
        'prior_path': str(prior_path), 'prior_sha256': sha256(prior_path),
        'skipped_exact_ranking_prefix': 256, 'scope': __doc__}
    return original_search(teacher, diagnostics)


base.diverse_candidates = remaining
base.search_flat_plate = search
base.main()
