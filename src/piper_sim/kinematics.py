"""URDF forward kinematics for offline inspection. No simulator dependency."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    x, y, z = a
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = (rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch)
                           @ rotation([1, 0, 0], roll))
    return result


def forward(urdf, positions, base=None):
    root = ET.parse(urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    def joint_position(name, seen=()):
        if name in seen:
            raise ValueError("Cyclic mimic joints")
        mimic = joints[name].find("mimic")
        if mimic is None:
            return positions.get(name, 0.)
        value = (float(mimic.get("multiplier", "1"))*joint_position(mimic.get("joint"), (*seen, name))
                 + float(mimic.get("offset", "0")))
        if name in positions and not np.isclose(positions[name], value, atol=1e-9):
            raise ValueError(f"{name} must follow its mimic joint")
        return value
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = {link.get("name") for link in root.findall("link")} - children
    transforms = {name: np.eye(4) if base is None else base.copy() for name in roots}
    pending = list(root.findall("joint"))
    while pending:
        count = len(pending)
        for joint in list(pending):
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            motion = np.eye(4)
            q = joint_position(joint.get("name"))
            axis = joint.find("axis")
            if joint.get("type") != "fixed":
                axis = np.fromstring(axis.get("xyz"), sep=" ")
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = axis*q
                else:
                    motion[:3, :3] = rotation(axis, q)
            transforms[joint.find("child").get("link")] = (
                transforms[parent] @ origin(joint.find("origin")) @ motion)
            pending.remove(joint)
        if len(pending) == count:
            raise ValueError("Disconnected or cyclic URDF")
    return transforms
