"""Piper articulation definition; independent of workcell and data collection."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from .config import ROOT
from .contracts import JOINT_NAMES


def piper_config(cfg, side):
    urdf = ROOT / cfg["robot"]["urdf"]
    if not urdf.exists():
        raise FileNotFoundError(f"{urdf}; run scripts/prepare_assets.py first")
    home = cfg["robot"]["home"]
    mount = cfg["robot"][side]
    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(urdf), usd_dir=str(urdf.parent / "usd"),
        usd_file_name="piper.usd", fix_base=True, root_link_name="base_link", merge_fixed_joints=True,
        self_collision=True, collider_type="convex_decomposition",
        # Lab 2.3.0 passes this field directly to importer.set_parse_mimic;
        # despite the field name, True is required to create PhysX mimic joints.
        convert_mimic_joints_to_normal_joints=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=2))
    return ArticulationCfg(
        prim_path=f"/World/{side}", spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(mount["position"]), rot=tuple(mount["rotation"]),
            joint_pos={**dict(zip(JOINT_NAMES, home[:6])),
                       "joint7": home[6]/2, "joint8": home[6]/2}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100., velocity_limit_sim=2., stiffness=400., damping=40.),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["joint7"],
                effort_limit_sim=20., velocity_limit_sim=.1, stiffness=4000., damping=100.),
            # The follower retains state for geometry/feedback but has no drive.
            "gripper_follower": ImplicitActuatorCfg(joint_names_expr=["joint8"],
                effort_limit_sim=0., velocity_limit_sim=.1, stiffness=0., damping=0.)})
