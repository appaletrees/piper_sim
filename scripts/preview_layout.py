#!/usr/bin/env python3
"""Offline geometric preview of configured URDFs; NOT an Isaac physics render."""
from pathlib import Path
import struct
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from piper_sim.config import ROOT, load
from piper_sim.contracts import JOINT_NAMES, ARMS
from piper_sim.kinematics import forward, origin
from piper_sim.transforms import load_camera_rig, pose_matrix


def stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    return np.frombuffer(raw, offset=84, count=count,
                         dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"]


def box(size, center):
    x, y, z = np.array(size)/2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])+center
    return [v[idx] for idx in [[0,1,2,3], [4,5,6,7], [0,1,5,4],
                               [2,3,7,6], [0,3,7,4], [1,2,6,5]]]


cfg = load()
rig = load_camera_rig(cfg)
urdf = ROOT / cfg["robot"]["urdf"]
fig = plt.figure(figsize=(12, 8), facecolor="#f5f7fa")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#f5f7fa")
for side, color in zip(ARMS, ["#cf864a", "#548caa"]):
    home = cfg["robot"]["home"]
    q = {**dict(zip(JOINT_NAMES, home[:6])), "joint7": home[6]/2, "joint8": home[6]/2}
    base = np.eye(4)
    base[:3, 3] = cfg["robot"][side]["position"]
    w, x, y, z = cfg["robot"][side]["rotation"]
    base[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    poses = forward(urdf, q, base)
    with np.load((ROOT/rig["mounts"][side]["usd"]).with_suffix(".npz")) as meshes:
        for i in range(len(meshes.files)//3):
            points = meshes[f"points_{i}"] @ poses["link6"][:3, :3].T + poses["link6"][:3, 3]
            ax.add_collection3d(Poly3DCollection(points[meshes[f"triangles_{i}"]],
                facecolors=meshes[f"color_{i}"], edgecolor="none", linewidth=0, shade=True))
    for link in ET.parse(urdf).getroot().findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            vertices = stl(mesh.get("filename"))
            transform = poses[link.get("name")] @ origin(visual.find("origin"))
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            ax.add_collection3d(Poly3DCollection(vertices, facecolors=color,
                                                 edgecolor="none", linewidth=0, shade=True))
    ax.text(*base[:3, 3] + [0, 0, .15], side, fontsize=10)
    print(side, "TCP:", poses["gripper_center"][:3, 3])
if cfg["scene"].get("geometry") == "cad":
    with np.load((ROOT/cfg["scene"]["cad_usd"]).with_suffix(".npz")) as meshes:
        for i in range(len(meshes.files)//3):
            vertices = meshes[f"points_{i}"][meshes[f"triangles_{i}"]]
            ax.add_collection3d(Poly3DCollection(vertices, facecolors=meshes[f"color_{i}"],
                                                edgecolor="none", linewidth=0, shade=True))
    primitives = [("object", "#d95146")]
else:
    primitives = [("table", "#b2bdc9"), ("mast", "#7d8b9a"), ("object", "#d95146")]
for name, color in primitives:
    part = cfg["scene"][name]
    ax.add_collection3d(Poly3DCollection(box(part["size"], part["position"]),
                                        facecolor=color, edgecolor="#667788", linewidth=.4))
mount = rig["cameras"]["overhead"]
cam = np.array(mount["position"])
ax.scatter(*cam, s=50, color="#292e40")
target = cam + pose_matrix(cam, mount["rotation_wxyz_ros"])[:3, 2]*.7
ax.plot(*np.array([cam, target]).T, "--", color="#8c86a8", linewidth=1)
ax.set(xlim=(-.2, .8), ylim=(-.8, .8), zlim=(0, 1.3),
       xlabel="X forward [m]", ylabel="Y left [m]", zlabel="Z up [m]")
ax.set_box_aspect([1, 1.6, 1.3])
ax.view_init(elev=24, azim=38)
ax.set_title("Dual Piper | 600 mm spacing | 1400 x 700 mm table\n"
             "Source CAD camera housings and mounts | offline geometry preview", pad=20)
output = ROOT / "outputs/layout_preview.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=150, bbox_inches="tight")
print(output)
