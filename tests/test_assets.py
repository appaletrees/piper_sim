from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from piper_sim.config import ROOT, load
from piper_sim.kinematics import forward


def test_prepared_fingers_open_outward_and_meshes_exist():
    cfg = load()
    urdf = ROOT / cfg["robot"]["urdf"]
    root = ET.parse(urdf).getroot()
    assert [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"] == [
        f"joint{i}" for i in range(1, 7)]
    for mesh in root.iter("mesh"):
        assert Path(mesh.get("filename")).is_file()
    shut = forward(urdf, {})
    opened = forward(urdf, {"joint7": .02, "joint8": .02})
    delta7 = opened["link7"][:3, 3]-shut["link7"][:3, 3]
    delta8 = opened["link8"][:3, 3]-shut["link8"][:3, 3]
    np.testing.assert_allclose(np.linalg.norm(delta7-delta8), .04, atol=1e-6)
