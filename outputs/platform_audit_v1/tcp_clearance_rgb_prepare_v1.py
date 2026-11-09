"""Freeze a full RGB/export regression after the three state trials."""
from argparse import Namespace
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

from piper_sim.collection_jobs import atomic_json, batch_identity, sha256
from piper_sim.config import ROOT
from piper_sim.lerobot_export import check_dependencies

base = ROOT/'outputs/platform_audit_v1'
plan_path, inputs_path = base/'tcp_clearance_rgb_plan_v1.json', base/'tcp_clearance_rgb_inputs_v1.json'
output = Path('/dev/shm/piper_tcp_clearance_rgb_v1')
if plan_path.exists() or inputs_path.exists() or output.exists():
    raise FileExistsError('Preserve earlier attempts and choose a new version')
state = Path('/dev/shm/piper_tcp_clearance_v1/worker_00/episode_000000_seed_1340830695/episode.h5')
prior = json.loads((base/'tcp_clearance_validation_v1.json').read_text())
assert sha256(state) == prior['runs'][0]['source_sha256']
prior_inputs = json.loads((base/'tcp_clearance_inputs_v1.json').read_text())
assert all(sha256(path) == digest for path, digest in prior_inputs['files'].items())
check_dependencies()
args = Namespace(config=str(ROOT/'configs/clean_table_tcp.yaml'),
    task_scene=str(ROOT/'configs/tasks/clean_table_rigid.yaml'),
    domain_randomization=str(ROOT/'configs/domain_randomization/clean_table.yaml'),
    collection_profile=str(ROOT/'configs/collection/clean_table_tcp_clearance.yaml'),
    acceptance_policy=str(ROOT/'configs/collection/quality_policy_tcp.yaml'),
    seed=1340830695, record=True, rgb=True, lerobot=True, grasp_mode='physical',
    viewer=False, video=False, objects=None)
identity = batch_identity(args)
for name in ('profile_collection.py', 'check_lerobot_reader.py'):
    path = ROOT/'scripts'/name
    identity['files'][str(path)] = sha256(path)
identity['versions'].update({k: importlib.metadata.version(k) for k in ('Pillow', 'pyarrow')})
identity['export_executables'] = {k: {'path': shutil.which(k), 'sha256': sha256(shutil.which(k))}
                                  for k in ('ffmpeg', 'ffprobe')}
free_disk, free_tmpfs = shutil.disk_usage(ROOT).free, shutil.disk_usage(output.parent).free
memory = dict((line.split(':')[0], int(line.split()[1])*1024)
              for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith(' kB'))
assert free_disk > 5*1024**3 and free_tmpfs > 20*1024**3 and memory['MemAvailable'] > 20*1024**3
command = [str(ROOT/'.venv-mujoco/bin/python'), str(ROOT/'scripts/profile_collection.py'),
    '--profile-output', str(output/'profile.json'), '--record', '--rgb', '--lerobot', '--seed', str(args.seed),
    '--output', str(output/'seed_1340830695')]
for key in ('config', 'task_scene', 'domain_randomization', 'collection_profile', 'acceptance_policy', 'grasp_mode'):
    command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
atomic_json(inputs_path, identity)
plan = {'purpose': 'Full synchronized RGB, independent TCP policy, training reader and LeRobot consumer regression',
    'seed': args.seed, 'selection': 'Known repaired three-cup regression; not a new holdout or population estimate',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'inputs_sha256': sha256(inputs_path), 'script_sha256': sha256(__file__), 'command': command,
    'raw_output': str(output), 'persistent_evidence': str(base/'tcp_clearance_rgb_evidence_v1'),
    'comparison_state_source': str(state), 'comparison_state_sha256': sha256(state),
    'free_disk_bytes_before': free_disk, 'free_tmpfs_bytes_before': free_tmpfs,
    'mem_available_bytes_before': memory['MemAvailable'],
    'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MUJOCO_GL')},
    'protocol': 'One complete physical task, all three native-resolution RGB streams at 30 Hz; no repeated preflight. '
                'Compare against the retained full state trajectory. No tuning, skipped images, speed editing, '
                'or acceptance relaxation. Keep failures and partial outputs.',
    'storage': 'Volatile tmpfs for full raw/export validation; not a durable production release. '
               'No existing raw data is deleted. Archive required before reboot.',
    'performance_scope': 'Profile full acquisition/audit/export on tmpfs; separate actual stage wall times '
                         'from production disk throughput. A single known regression cannot establish qualified population throughput.',
    'remaining_limitations': ['Uncalibrated camera/contact parameters and per-target finger material switching remain.',
                              'One accepted RGB episode would not demonstrate broad reliability or sim-to-real fidelity.']}
atomic_json(plan_path, plan)
output.mkdir()
print(json.dumps(plan, indent=2))
