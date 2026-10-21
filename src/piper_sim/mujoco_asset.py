"""Build MJCF from the same URDF and evaluated CAD meshes used by Isaac.

USD is read only during asset conversion. No Isaac, torch, CAN or simulator
runtime is imported here. Original dimensions and local frames are retained.
"""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT
from .kinematics import origin
from .transforms import load_camera_rig, transform_points


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).reshape(-1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_fingerprint(cfg):
    """Simulation settings only; machine-local SDK and URDF paths do not vary geometry."""
    data = {"robot": {k: cfg["robot"][k] for k in ("home", "left", "right", "urdf")},
            "physics_dt": cfg["runtime"]["physics_dt"], "decimation": cfg["runtime"]["decimation"],
            "scene": cfg["scene"], "cameras": cfg["cameras"], "appearance": cfg.get("appearance", {}),
            "substeps": cfg.get("mujoco", {}).get("substeps", 4),
            "optics_sha256": digest(ROOT/cfg["cameras"]["rgb_profiles"]),
            "materials_sha256": digest(ROOT/"configs/appearance.json")}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def resolve_model(cfg):
    """Use the portable model or a generated cache; rebuild changed source/settings."""
    for directory in (ROOT/"assets/mujoco", ROOT/"assets/generated/mujoco"):
        path = directory/"manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        if (manifest.get("configuration_sha256") != configuration_fingerprint(cfg)
                or manifest.get("builder_sha256") != digest(__file__)):
            continue
        if any((ROOT/name).is_file() and digest(ROOT/name) != expected
               for name, expected in manifest["inputs"].items()):
            continue
        model = directory/"workcell.xml"
        if digest(model) != manifest["model_sha256"]:
            raise ValueError(f"MJCF changed since export: {model}")
        for name, expected in manifest["mesh_sha256"].items():
            if digest(directory/"meshes"/name) != expected:
                raise ValueError(f"Mesh changed since export: {name}")
        if digest(directory/"kinematics.urdf") != manifest["kinematics_sha256"]:
            raise ValueError("Kinematic reference changed since export")
        return model, manifest
    try:
        return build_model(cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Settings differ from the portable MJCF. Prepare source assets with "
                                "scripts/install_mujoco.sh --rebuild-assets, then retry.") from exc


def quaternion(matrix):
    from pxr import Gf
    q = Gf.Matrix3d(*np.asarray(matrix).T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return numbers([q.GetReal(), *q.GetImaginary()])


def usd_triangles(stage, path):
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not len(counts) or np.any(counts != 3):
        raise ValueError(f"Expected a triangulated source CAD mesh: {path}")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    return transform_points(points, matrix), np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)


def build_model(cfg, output=None):
    """Create a self-contained MJCF + mesh directory and conversion manifest."""
    from pxr import Usd
    output = Path(output or ROOT/"assets/generated/mujoco").resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir = output/"meshes"
    mesh_dir.mkdir(exist_ok=True)
    urdf_path = ROOT/cfg["robot"]["urdf"]
    shutil.copy2(urdf_path.parent/"UPSTREAM_LICENSE", output/"UPSTREAM_LICENSE")
    rig = load_camera_rig(cfg)
    workcell_path = ROOT/cfg["scene"]["cad_usd"]
    source_manifest = json.loads(workcell_path.with_suffix(".json").read_text())
    if cfg["scene"].get("geometry") != "cad":
        raise ValueError("MuJoCo builder requires the authoritative CAD workcell")
    if source_manifest["sha256"] != rig["source_sha256"]:
        raise ValueError("Camera rig and workcell must use the same CAD revision")
    appearance = json.loads((ROOT/"configs/appearance.json").read_text())
    cad_gripper = cfg.get("appearance", {}).get("cad_gripper", False)
    gripper_path = ROOT/"assets/generated/cad_gripper/manifest.json"
    gripper = json.loads(gripper_path.read_text()) if cad_gripper else None
    if gripper and gripper["source_sha256"] != rig["source_sha256"]:
        raise ValueError("Gripper and camera CAD revisions differ")
    root = ET.Element("mujoco", model="dual_piper_cad")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true",
                  inertiafromgeom="false", fusestatic="false")
    # 120 Hz physics ticks are shared with Isaac; MuJoCo uses four smaller
    # internal integration steps per physics tick for stable stiff contacts.
    substeps = int(cfg.get("mujoco", {}).get("substeps", 4))
    if substeps < 1 or cfg["runtime"]["physics_dt"] <= 0:
        raise ValueError("Invalid MuJoCo integration interval")
    ET.SubElement(root, "option", timestep=str(cfg["runtime"]["physics_dt"]/substeps),
                  gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic",
                  iterations="100", tolerance="1e-9")
    ET.SubElement(root, "statistic", center="0.25 0 0.4", extent="2")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", azimuth="135", elevation="-28")
    ET.SubElement(visual, "map", znear="0.005", zfar="10")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", width="256", height="1536",
                  rgb1=".72 .78 .84", rgb2=".88 .9 .93")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true",
                  diffuse="0.8 0.8 0.8", ambient="0.35 0.35 0.35")
    ET.SubElement(world, "light", pos="2 -2 2", dir="-1 1 -1", directional="true",
                  diffuse=".45 .45 .45", castshadow="false")
    ET.SubElement(world, "geom", name="ground", type="plane", pos="0 0 -0.03", size="3 3 .05",
                  rgba=".25 .28 .32 1", friction="1 .005 .0001")
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    equality = ET.SubElement(root, "equality")
    tendon = ET.SubElement(root, "tendon")
    manifest = {"schema_version": 1, "source_cad_sha256": rig["source_sha256"],
                "units": "metres", "geometry_policy": "authoritative_source_CAD_no_visual_fitting",
                "urdf_sha256": digest(urdf_path), "rgb_profiles_sha256": rig["rgb_profiles_sha256"],
                "substeps": substeps, "camera_rig": rig, "cad_meshes": [],
                "collision_model": "individual CAD part convex hulls; URDF link convex hulls; exact tabletop box",
                "inertia_model": "source URDF; camera/bracket masses not identified",
                "inputs": {str(p.relative_to(ROOT)): digest(p) for p in
                           [urdf_path, workcell_path.with_suffix(".json"),
                            ROOT/cfg["cameras"]["rig_manifest"], *([gripper_path] if gripper else [])]},
                "robot_bodies": {}, "joint_names": [],
                "gripper_control": {"independent_dof_per_arm": 1, "target": "full opening in metres",
                    "actuator": "{side}/gripper", "coupling": "joint8=joint7; tendon=joint7+joint8"}}
    manifest["configuration_sha256"] = configuration_fingerprint(cfg)
    manifest["builder_sha256"] = digest(__file__)

    stages = {}
    def stage(path):
        path = Path(path).resolve()
        if path not in stages:
            stages[path] = Usd.Stage.Open(str(path))
            if stages[path] is None:
                raise ValueError(f"Cannot read CAD asset {path}")
            manifest["inputs"][str(path.relative_to(ROOT))] = digest(path)
        return stages[path]

    def add_mesh(parent, name, points, faces, material, collision=True, source=None):
        # OBJ retains the evaluated mesh coordinates. MuJoCo's internal mesh
        # centring is automatically represented in the compiled geom transform.
        path = mesh_dir/f"{name}.obj"
        with path.open("w") as stream:
            stream.writelines("v "+numbers(p)+"\n" for p in points)
            stream.writelines("f "+" ".join(str(int(i)+1) for i in f)+"\n" for f in faces)
        ET.SubElement(asset, "mesh", name=name, file=path.name, inertia="shell")
        ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name,
                      rgba=numbers([*material, 1]), contype="1" if collision else "0",
                      conaffinity="1" if collision else "0", group="0", friction="1 .005 .0001")
        manifest["cad_meshes"].append({"name": name, "parent": parent.get("name", "world"),
            "bounds_local_m": [points.min(0).tolist(), points.max(0).tolist()],
            "vertices": len(points), "triangles": len(faces), "source": source,
            "obj_sha256": digest(path)})

    cad_colors = {i: appearance["materials"][name]["diffuse_color"]
                  for name, indices in appearance["cad_parts"].items() for i in indices}
    remaining = appearance["materials"][appearance["remaining_cad_parts"]]["diffuse_color"]
    for part in source_manifest["parts"]:
        if part.get("target") != "static" or part.get("guide"):
            continue
        index = part["occurrence"]
        points, faces = usd_triangles(stage(workcell_path), part["usd_prim"])
        add_mesh(world, f"cad_static_{index}", points, faces, cad_colors.get(index, remaining),
                 collision=index != 0, source=part["usd_prim"])
        if index == 0:
            lo, hi = points.min(0), points.max(0)
            ET.SubElement(world, "geom", name="table_collision", type="box",
                pos=numbers((lo+hi)/2), size=numbers((hi-lo)/2), rgba="0 0 0 0", group="3",
                friction="1 .005 .0001")

    urdf = ET.parse(urdf_path).getroot()
    # A mesh-free URDF preserves the source kinematics for offline FK checks on
    # another machine, without the original absolute STL paths or an Isaac install.
    reference = ET.parse(urdf_path).getroot()
    for link in reference.findall("link"):
        for element in [*link.findall("visual"), *link.findall("collision")]:
            link.remove(element)
    ET.indent(reference)
    reference_path = output/"kinematics.urdf"
    ET.ElementTree(reference).write(reference_path, encoding="utf-8", xml_declaration=True)
    manifest["kinematics_sha256"] = digest(reference_path)
    links = {e.get("name"): e for e in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    copied = set()
    bodies = {}
    for side in ("left", "right"):
        mount = cfg["robot"][side]
        base = ET.SubElement(world, "body", name=f"{side}/base_link",
                             pos=numbers(mount["position"]), quat=numbers(mount["rotation"]))

        def add_link(link_name, body):
            name = f"{side}/{link_name}"
            bodies[name] = body
            manifest["robot_bodies"][name] = link_name
            link = links[link_name]
            inertial = link.find("inertial")
            if inertial is not None:
                t = origin(inertial.find("origin"))
                inertia = inertial.find("inertia")
                ixx, ixy, ixz, iyy, iyz, izz = [float(inertia.get(k)) for k in
                                              ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
                tensor = t[:3, :3] @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ t[:3, :3].T
                ET.SubElement(body, "inertial", mass=inertial.find("mass").get("value"),
                    pos=numbers(t[:3, 3]), fullinertia=numbers([
                        tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
            if not (cad_gripper and link_name in ("gripper_base", "link7", "link8")):
                for j, element in enumerate(link.findall("visual")):
                    mesh = element.find("geometry/mesh")
                    if mesh is None:
                        raise ValueError(f"Unsupported URDF visual geometry: {name}")
                    filename = Path(mesh.get("filename"))
                    mesh_name = "urdf_"+filename.stem
                    if mesh_name not in copied:
                        shutil.copy2(filename, mesh_dir/filename.name)
                        ET.SubElement(asset, "mesh", name=mesh_name, file=filename.name, inertia="shell")
                        copied.add(mesh_name)
                        manifest["inputs"][str(filename.relative_to(ROOT))] = digest(filename)
                    t = origin(element.find("origin"))
                    color = appearance["materials"][appearance["robot_links"][link_name]]["diffuse_color"]
                    ET.SubElement(body, "geom", name=f"{name}/mesh{j}", type="mesh", mesh=mesh_name,
                        pos=numbers(t[:3, 3]), quat=quaternion(t[:3, :3]), rgba=numbers([*color, 1]),
                        friction="1 .005 .0001")
            for joint in children.get(link_name, []):
                child_name = joint.find("child").get("link")
                t = origin(joint.find("origin"))
                child = ET.SubElement(body, "body", name=f"{side}/{child_name}",
                                      pos=numbers(t[:3, 3]), quat=quaternion(t[:3, :3]))
                if joint.get("type") != "fixed":
                    joint_name = f"{side}/{joint.get('name')}"
                    limit = joint.find("limit")
                    slide = joint.get("type") == "prismatic"
                    ET.SubElement(child, "joint", name=joint_name, type="slide" if slide else "hinge",
                        axis=joint.find("axis").get("xyz"), range=f"{limit.get('lower')} {limit.get('upper')}",
                        damping="1" if slide else "1.5", armature="0.005" if slide else "0.01")
                    if not slide:
                        ET.SubElement(actuator, "position", name=joint_name, joint=joint_name,
                            kp="400", kv="40", ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
                            forcerange="-100 100")
                    manifest["joint_names"].append(joint_name)
                # Explicit adjacent exclusions preserve the URDF joint's overlapping
                # bearing surfaces, including fixed gripper_base intermediates.
                ET.SubElement(contact, "exclude", body1=name, body2=f"{side}/{child_name}")
                add_link(child_name, child)

        add_link("base_link", base)
        mimic = urdf.find("joint[@name='joint8']/mimic")
        if (mimic is None or mimic.get("joint") != "joint7"
                or float(mimic.get("multiplier", "1")) != 1 or float(mimic.get("offset", "0")) != 0):
            raise ValueError("Expected joint8 to mimic joint7; rerun scripts/prepare_assets.py")
        ET.SubElement(equality, "joint", name=f"{side}/gripper_coupling",
            joint1=f"{side}/joint8", joint2=f"{side}/joint7", polycoef="0 1 0 0 0",
            solref="0.0042 1", solimp="0.999 0.9999 0.001")
        width = ET.SubElement(tendon, "fixed", name=f"{side}/gripper_width")
        for finger in ("joint7", "joint8"):
            ET.SubElement(width, "joint", joint=f"{side}/{finger}", coef="1")
        finger_limit = urdf.find("joint[@name='joint7']/limit")
        ET.SubElement(actuator, "position", name=f"{side}/gripper", tendon=f"{side}/gripper_width",
            kp="1000", kv="25", ctrlrange=numbers([0, 2*float(finger_limit.get("upper"))]),
            forcerange="-10 10")
        for part in source_manifest["parts"]:
            if part.get("target") != side:
                continue
            points, faces = usd_triangles(stage(ROOT/rig["mounts"][side]["usd"]), part["usd_prim"])
            i = part["occurrence"]
            add_mesh(bodies[f"{side}/link6"], f"cad_{side}_{i}", points, faces,
                     cad_colors.get(i, remaining), source=part["usd_prim"])
        if gripper:
            for link, record in gripper["arms"][side]["links"].items():
                for part in record["parts"]:
                    p = f"/CadGripper/part_{part['occurrence']:03d}"
                    mesh_stage = stage(ROOT/record["asset"])
                    points, faces = usd_triangles(mesh_stage, p)
                    from pxr import UsdGeom
                    color = list(UsdGeom.Mesh(mesh_stage.GetPrimAtPath(p)).GetDisplayColorAttr().Get()[0])
                    add_mesh(bodies[f"{side}/{link}"], f"gripper_{side}_{part['occurrence']}",
                             points, faces, color, source=p)

    for name, camera in rig["cameras"].items():
        from .transforms import pose_matrix
        t = pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
        rotation = t[:3, :3] @ np.diag([1., -1., -1.])
        parent = world if name == "overhead" else bodies[f"{name.split('_')[0]}/link6"]
        width, height = camera["rgb"]["resolution"]
        k = np.asarray(camera["rgb"]["intrinsic_matrix"])
        # MuJoCo's physical sensor dimensions let fx and fy remain independent.
        # ROS K indexes pixel centres; OpenGL uses raster boundaries and Y-up.
        # Include the half-pixel offset, verified by native/asymmetric RGB renders.
        ET.SubElement(parent, "camera", name=name, pos=numbers(t[:3, 3]), quat=quaternion(rotation),
            resolution=f"{width} {height}", sensorsize="1 1", focal=numbers([k[0, 0]/width, k[1, 1]/height]),
            principal=numbers([(width/2-k[0, 2]-.5)/width, (k[1, 2]+.5-height/2)/height]))
    obj = cfg["scene"]["object"]
    body = ET.SubElement(world, "body", name="object", pos=numbers(obj["position"]))
    ET.SubElement(body, "freejoint", name="object/free")
    size = np.asarray(obj["size"])
    inertia = .05/12*np.array([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
    ET.SubElement(body, "inertial", pos="0 0 0", mass=".05", diaginertia=numbers(inertia))
    ET.SubElement(body, "geom", name="object", type="box", size=numbers(size/2), rgba=".8 .15 .08 1",
                  friction="1 .005 .0001")
    home = cfg["robot"]["home"]
    positions = [*home[:6], home[6]/2, home[6]/2]*2
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=numbers([*positions, *obj["position"], 1, 0, 0, 0]),
                  ctrl=numbers(home*2))
    ET.indent(root)
    path = output/"workcell.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    manifest["model_sha256"] = digest(path)
    manifest["mesh_sha256"] = {p.name: digest(p) for p in sorted(mesh_dir.iterdir()) if p.is_file()}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return path, manifest
