#!/usr/bin/env python3
"""Cross-engine single-input gripper audit; pure simulation, no hardware."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from piper_sim.config import load
from piper_sim.contracts import Command


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("isaac", "mujoco"), required=True)
parser.add_argument("--output", type=Path, required=True)
select = argparse.ArgumentParser(add_help=False)
select.add_argument("--backend", required=True)
selected, _ = select.parse_known_args()
if selected.backend == "isaac":
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
app = AppLauncher(args).app if args.backend == "isaac" else None
backend = None
try:
    if args.backend == "isaac":
        from piper_sim.backends.isaac import IsaacBackend
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import PhysxSchema
        backend = IsaacBackend(load(), args.device)
        mimic = [str(p.GetPath()) for p in get_current_stage().Traverse()
                 if p.HasAPI(PhysxSchema.PhysxMimicJointAPI)]
        assert len(mimic) == 2, mimic
        for arm in backend.arms.values():
            assert bool((arm.actuators["gripper_follower"].stiffness == 0).all())
    else:
        from piper_sim.backends.mujoco import MujocoBackend
        backend = MujocoBackend(load())
        assert backend.model.nu == 14 and backend.model.neq == 2
        mimic = ["left/gripper_coupling", "right/gripper_coupling"]
    trials = []
    for sequence, width in enumerate((.01, .04, .07, .085, .095)):
        target = np.array(load()["robot"]["home"]*2)
        target[[6, 13]] = width
        backend.apply(Command(target, time.monotonic(), sequence))
        for _ in range(90):
            backend.step()
        trial = {"target_width_m": width}
        for side in ("left", "right"):
            if args.backend == "isaac":
                q = backend.arms[side].data.joint_pos[0, backend.indices[side][-2:]].cpu().numpy()
            else:
                q = backend.data.qpos[backend.indices[side]["qpos"][-2:]]
            difference, error = float(abs(q[0]-q[1])), float(abs(q.sum()-width))
            assert difference < 1e-4, (side, q)
            assert error < .0015, (side, q, width)
            trial[side] = {"finger_positions_m": q.tolist(), "coupling_error_m": difference,
                           "width_error_m": error}
        trials.append(trial)
    result = {"passed": True, "backend": args.backend, "independent_gripper_dof_per_arm": 1,
              "couplings": mimic, "trials": trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
finally:
    if backend:
        backend.close()
    if app:
        app.close(wait_for_replicator=False)
