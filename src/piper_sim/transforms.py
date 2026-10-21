"""Small, engine-independent rigid transform utilities (SI units, wxyz quaternions)."""
import numpy as np


def pose_matrix(position, quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = position
    return result


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def load_camera_rig(cfg):
    import json
    from .config import ROOT
    path = ROOT / cfg["cameras"]["rig_manifest"]
    if not path.is_file():
        raise FileNotFoundError(f"{path}; run scripts/convert_workcell_cad.py")
    return json.loads(path.read_text())
