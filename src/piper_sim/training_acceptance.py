"""Shared read-only training gate; no renderer, encoder or simulator imports."""
import json
from pathlib import Path

import h5py


def require_self_contained(f):
    """Require the project's hard-link / same-file VDS storage layout.

    A whole-file checksum cannot protect values stored in other files. Soft
    aliases are outside this supported layout, even when their target is local.
    """
    def check(name, obj):
        if isinstance(obj, h5py.Group):
            for key in obj:
                if not isinstance(obj.get(key, getlink=True), h5py.HardLink):
                    raise ValueError('Unsupported HDF5 link (require hard links): '+obj.name+'/'+key)
        elif isinstance(obj, h5py.Dataset):
            if obj.external:
                raise ValueError('External HDF5 dataset storage: '+name)
            if obj.is_virtual and any(source.file_name != '.' for source in obj.virtual_sources()):
                raise ValueError('External HDF5 virtual source: '+name)
    check('/', f)
    f.visititems(check)


def verify_training_acceptance(f, path, *, policy=None):
    """Verify the selected policy and checksum on the same open file being read.

    Returns the embedded report and final file digest. A caller streaming data
    must keep this handle open until reading finishes, retaining its read lock.
    """
    from .collection_audit import AUDIT_VERSION, load_acceptance_policy, _descriptor_sha256, _same_open_file

    path = Path(path).resolve()
    require_self_contained(f)
    audit = json.loads(f.attrs.get('dataset_audit', '{}'))
    if not (f.attrs.get('training_ready', False) and f.attrs.get('accepted', False)
            and f.attrs.get('success', False) and audit.get('training_ready')
            and audit.get('passed') and not audit.get('violations')):
        raise ValueError('Training data requires a passed RGB dataset audit: '+str(path))
    sidecar = f.attrs.get('dataset_audit_sidecar', '')
    if not sidecar or Path(sidecar).name != sidecar:
        raise ValueError('Missing portable dataset audit sidecar: '+str(path))
    evidence = json.loads(path.with_name(sidecar).read_text())
    audited_sha256 = evidence.get('source_sha256')
    descriptor = f.id.get_vfd_handle()
    if (not evidence.get('training_ready') or not evidence.get('passed') or evidence.get('violations')
            or audited_sha256 != _descriptor_sha256(descriptor) or not _same_open_file(path, descriptor)):
        raise ValueError('Episode changed after its dataset audit: '+str(path))
    _, expected_policy = load_acceptance_policy(policy)
    if (audit.get('schema_version') != AUDIT_VERSION or
            audit.get('acceptance_policy', {}).get('sha256') != expected_policy['sha256']):
        raise ValueError('Re-audit this episode with the selected production acceptance policy: '+str(path))
    execution = audit.get('checks', {}).get('position_execution', {})
    if (execution.get('passed') is not True or execution.get('violations') != []
            or execution.get('position_only_execution_supported') is not True
            or execution.get('checked_intervals') != int(f.attrs['num_transitions'])):
        raise ValueError('Training data requires complete position execution evidence: '+str(path))
    if {k: v for k, v in evidence.items() if k != 'source_sha256'} != audit:
        raise ValueError('Embedded and detached dataset audits disagree: '+str(path))
    return audit, audited_sha256
