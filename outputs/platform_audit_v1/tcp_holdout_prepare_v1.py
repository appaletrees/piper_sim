"""Predeclare six new state-only TCP trials without inspecting their layouts."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from piper_sim.collection_jobs import atomic_json, batch_identity, sha256
from piper_sim.config import ROOT

base = ROOT/'outputs/platform_audit_v1'
plan_path = base/'tcp_holdout_plan_v1.json'
identity_path = base/'tcp_holdout_inputs_v1.json'
output = Path('/dev/shm/piper_tcp_holdout_v1')
if plan_path.exists() or identity_path.exists() or output.exists():
    raise FileExistsError('Keep existing attempts; select a new version explicitly')
namespace = 'piper/platform_audit/tcp_frozen_holdout/v1'
seeds = [int.from_bytes(hashlib.sha256(f'{namespace}/{index}'.encode()).digest()[:4], 'big')
         for index in range(6)]
assert len(set(seeds)) == 6
# Search existing recorded reports/plans/analysis, including literal seed use in
# scripts. Reject any occurrence rather than claiming it is a new trial.
matches = subprocess.run(['rg', '-l', '--hidden', '--no-ignore', '-g', '*.json', '-g', '*.jsonl',
                          '-g', '*.py', '|'.join(map(str, seeds)),
                          str(ROOT/'outputs'), str(ROOT/'scripts'), str(ROOT/'configs')],
                         capture_output=True, text=True)
assert matches.returncode == 1, matches.stdout+matches.stderr
args = Namespace(output=output, resume=False, workers=3, seeds=seeds,
                 config=str(ROOT/'configs/clean_table_tcp.yaml'),
                 task_scene=str(ROOT/'configs/tasks/clean_table_rigid.yaml'),
                 domain_randomization=str(ROOT/'configs/domain_randomization/clean_table.yaml'),
                 collection_profile=str(ROOT/'configs/collection/clean_table_tcp.yaml'),
                 acceptance_policy=str(ROOT/'configs/collection/quality_policy_tcp.yaml'),
                 grasp_mode='physical', states_only=True, skip_preflight=False,
                 video=False, lerobot=False, timeout_s=7200., nominal_fraction=0., validation_fraction=.2)
identity = batch_identity(args)
parallel = ROOT/'scripts/collect_parallel_clean_table.py'
identity['files'][str(parallel)] = sha256(parallel)
atomic_json(identity_path, identity)
plan = {'namespace': namespace, 'selection': 'first six SHA256(namespace/index) uint32 big-endian seeds, no layout inspection',
        'seeds': seeds, 'prior_literal_matches': [], 'workers': args.workers,
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'inputs_sha256': sha256(identity_path), 'analysis_sha256': sha256(__file__),
        'config': args.config, 'collection_profile': args.collection_profile,
        'acceptance_policy': args.acceptance_policy,
        'original_policy_sha256': sha256(ROOT/'configs/collection/quality_policy.yaml'),
        'protocol': 'Complete physical state trials, retain all outcomes, no parameter changes until all six finish. Original quality and 300 s duration limits retained.',
        'raw_output': str(output), 'raw_storage': 'volatile tmpfs; retained until archival capacity is available; not a durable training release',
        'persistent_evidence': str(base/'tcp_holdout_evidence_v1'),
        'free_disk_bytes_before': shutil.disk_usage(ROOT).free,
        'free_tmpfs_bytes_before': shutil.disk_usage(output.parent).free,
        'training_ready': False,
        'limitations': ['Six diagnostic layouts do not certify population success, RGB acceptance, or hardware calibration.',
                       'Temporary raw files can be lost on reboot; archive remains required. No prior recording is removed.']}
atomic_json(plan_path, plan)
print(json.dumps(plan, indent=2))
