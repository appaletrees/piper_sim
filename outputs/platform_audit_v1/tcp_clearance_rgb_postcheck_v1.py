"""Sequential checks of a completed RGB acquisition; preserve every failure/log."""
import json
import os
from pathlib import Path
import subprocess
import time
from piper_sim.collection_jobs import atomic_json, sha256

repo = Path(__file__).resolve().parents[2]
base = repo/'outputs/platform_audit_v1'
raw = Path('/dev/shm/piper_tcp_clearance_rgb_v1')
source = raw/'seed_1340830695/episode.h5'
export = source.parent/'lerobot_v2'
output = base/'tcp_clearance_rgb_postcheck_v1.json'
assert not output.exists()
profile = json.loads((raw/'profile.json').read_text()); assert profile['returncode'] == 0
policy = repo/'configs/collection/quality_policy_tcp.yaml'
python = str(repo/'.venv-mujoco/bin/python')
target = lambda name: str(base/f'tcp_clearance_rgb_{name}_v1.json')
commands = [
 ('reader', [python, str(base/'tcp_clearance_rgb_validate_v1.py')], base/'tcp_clearance_rgb_validation_v1.json'),
 ('consumer', [str(repo/'.venv-lerobot-audit/bin/python'), 'scripts/check_lerobot_reader.py',
  '--source', str(source), '--dataset', str(export), '--output', target('consumer')], Path(target('consumer'))),
 ('camera_fk', [python, 'scripts/audit_camera_kinematics.py', str(source), '--output', target('camera_fk')], Path(target('camera_fk'))),
 ('video_audit', [python, 'scripts/audit_exported_video.py', '--source', str(source), '--dataset', str(export),
  '--policy', str(policy), '--output', target('video_audit')], Path(target('video_audit'))),
]
files = [Path(__file__), base/'tcp_clearance_rgb_validate_v1.py', repo/'scripts/check_lerobot_reader.py',
         repo/'scripts/audit_camera_kinematics.py', repo/'scripts/audit_exported_video.py', policy]
identity = {str(p): sha256(p) for p in files}
before = sha256(source)
export_before = {str(p.relative_to(export)): sha256(p) for p in export.rglob('*') if p.is_file()}
result = dict(status='running', source=str(source), source_sha256=before, export_checksums=export_before,
              checker_input_hashes=identity, steps=[], training_ready=None, durable_release=False)
atomic_json(output, result)
env = dict(os.environ, PYTHONPATH=str(repo/'src'), MUJOCO_GL='egl', HF_HUB_OFFLINE='1',
           HF_DATASETS_OFFLINE='1', HF_HOME=str(raw/'hf-cache'), HF_DATASETS_CACHE=str(raw/'hf-cache/datasets'))
start = time.perf_counter()
try:
    for name, command, report in commands:
        assert not report.exists(), report
        log = raw/f'postcheck_{name}.log'; assert not log.exists(), log
        print('Starting '+name, flush=True)
        begin = time.perf_counter()
        with log.open('x') as stream:
            completed = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(name=name, command=command, returncode=completed.returncode, wall_seconds=time.perf_counter()-begin,
                     log=str(log), log_sha256=sha256(log), report=str(report))
        if report.is_file(): entry['report_sha256'] = sha256(report)
        result['steps'].append(entry); atomic_json(output, result)
        print(json.dumps(entry), flush=True)
        if completed.returncode: raise RuntimeError(name+' failed')
    assert all(sha256(p) == digest for p, digest in identity.items())
    assert sha256(source) == before
    assert {str(p.relative_to(export)): sha256(p) for p in export.rglob('*') if p.is_file()} == export_before
    frozen = json.loads((base/'tcp_clearance_rgb_inputs_v1.json').read_text())
    assert all(sha256(p) == digest for p, digest in frozen['files'].items())
    result.update(status='all_checks_passed', source_and_export_unchanged=True, frozen_input_files_unchanged=len(frozen['files']))
except BaseException as error:
    result.update(status='failed_postcheck', error=repr(error))
    raise
finally:
    result['wall_seconds'] = time.perf_counter()-start
    atomic_json(output, result)
