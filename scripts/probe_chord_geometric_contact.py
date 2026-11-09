#!/usr/bin/env python3
"""Use first bilateral geometry, not scratch solver forces, in chord preflight.

The isolated closure scan holds the plate pose fixed. Scratch mj_forward forces
cannot identify its first physical collision: the plate can balance on the
table and leave intersecting finger contacts unloaded. Begin necessary force
screening at the first bilateral geometric contact instead. This is still a
fixed-pose approximation; it is not a physical grasp or a no-solution proof.
"""
from piper_sim.collection_jobs import sha256
import probe_plate_chord as base
import probe_chord_capture_feasibility as capture


def geometric_fingers(backend, side, name):
    model, data = backend.model, backend.data
    target = model.body('task/'+name).id
    fingers = {model.body(side+'/link'+str(i)).id: str(i) for i in (7, 8)}
    touched = set()
    for contact in data.contact:
        if contact.dist > 0.:
            continue
        bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
        if target in bodies:
            touched.update(fingers[b] for b in bodies if b in fingers)
    return sorted(touched)


def main():
    original, publish = base.finger_contacts, base.atomic_json
    profile = {'scope': __doc__, 'script_sha256': sha256(__file__),
               'contact_distance_max_m': 0., 'normal_force_threshold_used': False,
               'aperture_step_m': .0005,
               'production_contact_detector_changed': False}
    try:
        base.finger_contacts = geometric_fingers
        base.atomic_json = lambda path, value: publish(path, {
            **value, 'first_geometric_contact_screen': profile})
        return capture.main()
    finally:
        base.finger_contacts, base.atomic_json = original, publish


if __name__ == '__main__':
    main()
