"""Durable batch progress. Resume never overwrites a recording or trusts a stale gate."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .config import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def batch_identity(args):
    """Hash code, configs and actual assets; do not rely on Git clean status."""
    options = {k: v for k, v in vars(args).items() if k not in ('output', 'resume', 'timeout_s')}
    files = [p for p in (ROOT/'src/piper_sim').rglob('*.py') if 'rl' not in p.parts]
    files += [ROOT/'scripts'/name for name in ('collect_clean_table.py', 'demo_clean_table.py')]
    files += [p for p in (ROOT/'configs').rglob('*') if p.is_file() and 'rl' not in p.parts]
    files += [p for p in (ROOT/'assets').rglob('*') if p.is_file()]
    files += [Path(getattr(args, name)).resolve() for name in
              ('config', 'task_scene', 'domain_randomization', 'collection_profile')]
    if getattr(args, 'acceptance_policy', None):
        files.append(Path(args.acceptance_policy).resolve())
    fingerprints = {str(p.resolve()): sha256(p) for p in sorted(set(files))}
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'h5py')}
    return {'schema_version': 'piper.collection_batch.v1', 'options': options,
            'files': fingerprints, 'versions': versions}


def validate_completed(result, *, policy=None):
    """Recheck accepted provenance before reusing a completed job on resume."""
    if result['accepted']:
        from .lerobot_export import _inspect
        _inspect(Path(result['path']), require_quality=True, policy=policy)
    elif result.get('diagnostic_passed'):
        import h5py
        from .collection_audit import AUDIT_VERSION, load_acceptance_policy
        path = Path(result['path'])
        evidence = json.loads(Path(result['audit']).read_text())
        _, policy = load_acceptance_policy(policy)
        if (not evidence.get('passed') or evidence.get('schema_version') != AUDIT_VERSION or
                evidence.get('source_sha256') != sha256(path) or
                evidence.get('acceptance_policy', {}).get('sha256') != policy['sha256']):
            raise ValueError('Diagnostic source or policy changed: '+str(path))
        with h5py.File(path, 'r') as f:
            from .training_acceptance import require_self_contained
            require_self_contained(f)
            embedded = json.loads(f.attrs.get('dataset_audit', '{}'))
            if not f.attrs.get('complete') or embedded != {k: v for k, v in evidence.items() if k != 'source_sha256'}:
                raise ValueError('Diagnostic audit mismatch: '+str(path))


class BatchJournal:
    def __init__(self, root, identity, *, resume=False):
        self.root = Path(root)
        self.identity = identity
        self.resume = resume
        self.lock = None

    def __enter__(self):
        if self.resume:
            if not (self.root/'run.json').is_file():
                raise ValueError('No resumable run.json; historical batches require a new output directory')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self.lock = (self.root/'.collection.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                if json.loads((self.root/'run.json').read_text()) != self.identity:
                    raise ValueError('Batch code, configuration, assets or dependencies changed; use a new run directory')
            else:
                atomic_json(self.root/'run.json', self.identity)
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *_):
        self.lock.close()

    def completed(self, index):
        path = self.root/'jobs'/f'{index:06d}.json'
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        validate_completed(result, policy=self.identity.get('options', {}).get('acceptance_policy'))
        return result

    def output_for(self, index, seed):
        base = f'episode_{index:06d}_seed_{seed}'
        name, attempt = base, 1
        while any(self.root.glob(name+'*')):
            attempt += 1
            name = base+f'_attempt_{attempt:03d}'
        return self.root/name

    def save(self, index, result):
        atomic_json(self.root/'jobs'/f'{index:06d}.json', result)

    def publish(self, results, *, validation_fraction):
        for category in ('accepted', 'diagnostics', 'rejected', 'unverified'):
            atomic_text(self.root/(category+'.jsonl'), ''.join(
                json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in results if r['category'] == category))
        seconds = sum(sum(r.get('phase_wall_seconds', {}).values()) for r in results)
        accepted = sum(r['accepted'] for r in results)
        summary = {'attempted': len(results), 'accepted': accepted,
            'diagnostic_passed': sum(r['diagnostic_passed'] for r in results),
            'rejected': sum(r['category'] == 'rejected' for r in results),
            'unverified': sum(r['category'] == 'unverified' for r in results),
            'completed_phase_wall_seconds': seconds,
            'accepted_per_completed_phase_hour': accepted*3600/seconds if seconds else 0.,
            'episodes': results, 'validation_fraction': validation_fraction,
            'note': 'Phase time includes preflight and closed-file audits; excludes interrupted uncommitted attempts, '
                    'resume validation, batch export and idle time. Not an end-to-end throughput benchmark. '
                    'Finite seeds do not establish distribution-wide success.'}
        atomic_json(self.root/'summary.json', summary)
        return summary
