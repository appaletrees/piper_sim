"""Policy retiming changes position commands, never the fixed backend servo."""
from types import SimpleNamespace

from piper_sim.experts.clean_table import CleanTableTeacher


def test_gentle_separation_limits_apply_only_to_plate_lift():
    t=CleanTableTeacher.__new__(CleanTableTeacher)
    t.motion={'joint_velocity_rad_s':1.,'joint_acceleration_rad_s2':2.5,'joint_jerk_rad_s3':20.,
              'gripper_velocity_m_s':.06,'gripper_acceleration_m_s2':.18,'gripper_jerk_m_s3':1.5}
    t.pacing={'continuous_paths':True,
              'rim_load_limits':{'joint_velocity_rad_s':.75},
              'rim_lift_limits':{'joint_velocity_rad_s':.25,'joint_acceleration_rad_s2':.6,'joint_jerk_rad_s3':4.}}
    t.b=SimpleNamespace(task=SimpleNamespace(specs={'plate':{'kind':'plate'},'dish':{'kind':'side_dish'}}))
    original=t.motion.copy()
    t.carried=('right','plate');t.phase='lift'
    gentle=t.motion_limits()
    assert gentle['velocity'][7:13]==[.25]*6
    assert gentle['acceleration'][7:13]==[.6]*6
    assert gentle['jerk'][7:13]==[4.]*6
    assert gentle['velocity'][13]==.06
    t.phase='transfer'
    assert t.motion_limits()['velocity'][7:13]==[.75]*6
    t.carried=('right','dish');t.phase='lift'
    assert t.motion_limits()['velocity'][7:13]==[.75]*6
    t.carried=None
    assert t.motion_limits()['velocity'][7:13]==[1.]*6
    assert t.motion==original
