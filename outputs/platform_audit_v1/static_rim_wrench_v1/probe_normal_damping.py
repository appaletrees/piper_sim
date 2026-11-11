"""Synthetic normal-contact parameter check; not robot calibration, replay or training data."""
from pathlib import Path
import hashlib,json
import mujoco,numpy as np,yaml

rows=[]
for source in ('configs/physics/mujoco_fixed_fingers.yaml','configs/physics/mujoco_damped_fingers.yaml'):
 p=Path(source);c=yaml.safe_load(p.read_text());ref=' '.join(map(str,c['solref']));imp=' '.join(map(str,c['solimp']));values=[]
 xml=f'''<mujoco><option timestep="0.0010416666666666667" solver="Newton" cone="elliptic" iterations="100"/><worldbody><geom type="plane" size="1 1 .1" condim="1"/><body pos="0 0 .019"><freejoint/><geom type="sphere" size=".02" mass=".1" condim="1" priority="1" solref="{ref}" solimp="{imp}"/></body></worldbody></mujoco>'''
 model=mujoco.MjModel.from_xml_string(xml)
 for vz in (-.01,0.,.01):
  data=mujoco.MjData(model);data.qvel[2]=vz;mujoco.mj_forward(model,data)
  assert data.ncon==1 and data.nefc==1
  force=np.zeros(6);mujoco.mj_contactForce(model,data,0,force)
  values.append({'synthetic_velocity_z_m_s':vz,'distance_m':float(data.contact[0].dist),'normal_force_N':float(force[0]),'reference_acceleration_m_s2':float(data.efc_aref[0])})
 rows.append({'profile':source,'profile_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'solref':c['solref'],'solimp':c['solimp'],'samples':values})
zero=[r['samples'][1]['normal_force_N'] for r in rows];slopes=[(r['samples'][0]['normal_force_N']-r['samples'][2]['normal_force_N'])/.02 for r in rows]
assert np.isclose(zero[0],zero[1],rtol=0,atol=1e-9)
assert np.isclose(slopes[1]/slopes[0],2.,rtol=0,atol=1e-9)
out={'scope':__doc__,'training_ready':False,'mujoco_version':mujoco.__version__,'same_zero_velocity_contact_force':True,'normal_force_velocity_slope_ratio':slopes[1]/slopes[0],'profiles':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};Path('outputs/platform_audit_v1/static_rim_wrench_v1/normal_damping_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
