"""Freeze known regressions for the optional neighbor-clearance candidate."""
from argparse import Namespace
import json
from pathlib import Path
import shutil
import subprocess

from piper_sim.collection_jobs import atomic_json, batch_identity, sha256
from piper_sim.config import ROOT

base = ROOT/'outputs/platform_audit_v1'
plan_path, identity_path = base/'tcp_clearance_plan_v1.json', base/'tcp_clearance_inputs_v1.json'
output = Path('/dev/shm/piper_tcp_clearance_v1')
if plan_path.exists() or identity_path.exists() or output.exists():
    raise FileExistsError('Keep every previous attempt and select a new version')
seeds = [1340830695, 3074758446, 1990188034]
args = Namespace(output=output, resume=False, workers=3, seeds=seeds,
    config=str(ROOT/'configs/clean_table_tcp.yaml'),
    task_scene=str(ROOT/'configs/tasks/clean_table_rigid.yaml'),
    domain_randomization=str(ROOT/'configs/domain_randomization/clean_table.yaml'),
    collection_profile=str(ROOT/'configs/collection/clean_table_tcp_clearance.yaml'),
    acceptance_policy=str(ROOT/'configs/collection/quality_policy_tcp.yaml'),
    grasp_mode='physical', states_only=True, skip_preflight=False, video=False, lerobot=False,
    timeout_s=7200., nominal_fraction=0., validation_fraction=.2)
identity = batch_identity(args)
identity['files'][str(ROOT/'scripts/collect_parallel_clean_table.py')] = sha256(ROOT/'scripts/collect_parallel_clean_table.py')
atomic_json(identity_path, identity)
plan = {'seeds': seeds, 'workers': 3,
    'selection': 'Known regression: rejected neighboring-stick grasp, slowest prior three-cup pass, and a two-cup pass. Not new holdouts.',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'inputs_sha256': sha256(identity_path), 'analysis_sha256': sha256(__file__),
    'config': args.config, 'collection_profile': args.collection_profile, 'acceptance_policy': args.acceptance_policy,
    'original_policy_sha256': sha256(ROOT/'configs/collection/quality_policy.yaml'),
    'protocol': 'Full episodes from reset; no tuning during this cohort; original dynamics, materials and all quality limits retained.',
    'raw_output': str(output), 'raw_storage': 'volatile tmpfs; archival required; not a durable training release',
    'persistent_evidence': str(base/'tcp_clearance_evidence_v1'),
    'free_disk_bytes_before': shutil.disk_usage(ROOT).free,
    'free_tmpfs_bytes_before': shutil.disk_usage(output.parent).free,
    'training_ready': False,
    'limitations': ['Development regressions are not an independent reliability estimate.',
        'No RGB is recorded; accepted training episodes remain zero.',
        'Per-target finger material switching remains present and uncalibrated.',
        'Neighbor margin is a sampled geometric budget, not continuous-time or hardware certification.']}
atomic_json(plan_path, plan)
print(json.dumps(plan, indent=2))
