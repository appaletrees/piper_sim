"""Freeze the optional three-worker RGB regression before full acquisition."""
from argparse import Namespace
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

from piper_sim.collection_jobs import atomic_json,batch_identity,sha256
from piper_sim.config import ROOT,load
from piper_sim.lerobot_export import check_dependencies
from piper_sim.mujoco_asset import configuration_fingerprint

base=ROOT/'outputs/platform_audit_v1'; raw=Path('/dev/shm/piper_sensor_workers_rgb_v1')
plan_path=base/'sensor_workers_rgb_plan_v1.json'; inputs_path=base/'sensor_workers_rgb_inputs_v1.json'
assert not any(p.exists() for p in (raw,plan_path,inputs_path))
prior=json.loads((base/'tcp_clearance_rgb_summary_v1.json').read_text())
source=Path(prior['source']); assert sha256(source)==prior['source_sha256']
assert json.loads((base/'sensor_workers_gl_check_v2.json').read_text())['different_channel_values']==0
previous=json.loads((base/'tcp_clearance_rgb_inputs_v1.json').read_text())
changed={p for p,d in previous['files'].items() if sha256(p)!=d}
expected={str(ROOT/p) for p in ('src/piper_sim/backends/mujoco.py','src/piper_sim/backends/isaac.py',
                               'src/piper_sim/randomization/__init__.py')}
assert changed==expected,changed
old=load(ROOT/'configs/clean_table_tcp.yaml'); new=load(ROOT/'configs/clean_table_tcp_sensor_workers.yaml')
assert configuration_fingerprint(old)==configuration_fingerprint(new)
assert new['runtime'].pop('sensor_postprocess_workers')==3
old.pop('_config_path'); new.pop('_config_path'); assert old==new
check_dependencies()
args=Namespace(config=str(ROOT/'configs/clean_table_tcp_sensor_workers.yaml'),
 task_scene=str(ROOT/'configs/tasks/clean_table_rigid.yaml'),
 domain_randomization=str(ROOT/'configs/domain_randomization/clean_table.yaml'),
 collection_profile=str(ROOT/'configs/collection/clean_table_tcp_clearance.yaml'),
 acceptance_policy=str(ROOT/'configs/collection/quality_policy_tcp.yaml'),seed=1340830695,
 record=True,rgb=True,lerobot=True,grasp_mode='physical',viewer=False,video=False,objects=None)
identity=batch_identity(args)
for name in ('profile_collection.py','check_lerobot_reader.py','audit_exported_video.py'):
 path=ROOT/'scripts'/name; identity['files'][str(path)]=sha256(path)
identity['versions'].update({k:importlib.metadata.version(k) for k in ('Pillow','pyarrow')})
identity['export_executables']={k:{'path':shutil.which(k),'sha256':sha256(shutil.which(k))} for k in ('ffmpeg','ffprobe')}
free_tmpfs=shutil.disk_usage(raw.parent).free; free_root=shutil.disk_usage(ROOT).free
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith(' kB')}
allowance=int(1.2*(prior['source_bytes']+prior['export_bytes']))+2*1024**3
assert free_tmpfs>allowance and mem['MemAvailable']>18*1024**3
# No new raw files on root: only code and bounded small evidence. Its initial
# 5-GiB planning target has already drifted below that on the shared host.
assert free_root>4*1024**3
command=[str(ROOT/'.venv-mujoco/bin/python'),str(ROOT/'scripts/profile_collection.py'),'--profile-output',str(raw/'profile.json'),
 '--record','--rgb','--lerobot','--seed',str(args.seed),'--output',str(raw/'seed_1340830695')]
for key in ('config','task_scene','domain_randomization','collection_profile','acceptance_policy','grasp_mode'):
 command+=['--'+key.replace('_','-'),str(getattr(args,key))]
atomic_json(inputs_path,identity)
plan=dict(purpose='Complete known-scene RGB regression with three synchronous CPU camera workers',
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 source_worktree_changes='Uncommitted candidate frozen by per-file checksums; unrelated PPO changes excluded from this work.',
 seed=args.seed,command=command,raw_output=str(raw),inputs_sha256=sha256(inputs_path),script_sha256=sha256(__file__),
 comparison_rgb_source=str(source),comparison_rgb_sha256=prior['source_sha256'],
 comparison_summary_sha256=sha256(base/'tcp_clearance_rgb_summary_v1.json'),
 changed_existing_baseline_sources=sorted(changed),unchanged_model_configuration_fingerprint=configuration_fingerprint(old),
 free_tmpfs_bytes=free_tmpfs,free_root_bytes=free_root,mem_available_bytes=mem['MemAvailable'],
 output_allowance_bytes=allowance,maximum_new_persistent_evidence_bytes=2*1024**2,
 environment={k:os.environ.get(k) for k in ('MUJOCO_GL','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
 protocol='Same full task, seed, 30-Hz recording, native 640x480 RGB, physical teacher, timing and independent TCP policy. '
 'No preflight rerun, frame skipping, threshold relaxation, model changes or playback acceleration. Wait for every camera before stepping.',
 performance_scope='Compare with the retained prior serial run, acknowledging different shared-host execution windows and its early test interference. '
 'No new paired population-throughput or 2x full-pipeline claim.',
 storage='Raw and export remain on volatile tmpfs; preserve all failures and existing raw sources. Archival is required before reboot.',
 remaining_limitations=prior['limitations'])
atomic_json(plan_path,plan); raw.mkdir()
print(json.dumps(dict(raw_output=str(raw),frozen_files=len(identity['files']),free_tmpfs_bytes=free_tmpfs,output_allowance_bytes=allowance)))
