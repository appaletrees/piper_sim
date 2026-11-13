"""Storage migration must preserve bytes, canonical paths and live recordings."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from piper_sim.config import ROOT

spec=importlib.util.spec_from_file_location('nvme',ROOT/'scripts/collection_nvme.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)


def episode(tmp_path):
    run=tmp_path/'run';fast=tmp_path/'fast';fast.mkdir()
    for category in ('running','success','failed'):(run/category).mkdir(parents=True)
    p=n.create_folder(run,fast,'episode_000001_seed_2')
    (p/'recording').mkdir();(p/'recording/episode.h5').write_bytes(b'original RGB and joint data')
    (p/'process.json').write_text(json.dumps({'pid':999999999,'proc_start_ticks':'none'}))
    final=run/'success'/p.name
    (p/'result.json').write_text(json.dumps({'category':'success','directory':str(final)}))
    p.rename(final)
    return run,fast,final,tmp_path/'archive'


def test_archive_preserves_recording_bytes_and_stable_manifest_paths(tmp_path):
    run,fast,link,archive=episode(tmp_path)
    before=n.tree_manifest(link)
    assert n.archive_one(link,fast,archive)
    assert link.is_symlink() and link.resolve()==archive/'success'/link.name
    assert n.tree_manifest(link)==before
    assert not (fast/link.name).exists()
    assert json.loads((archive/'receipts'/(link.name+'.json')).read_text())['files']==before
    assert not n.archive_one(link,fast,archive)


def test_corrupt_copy_never_switches_or_deletes_original(tmp_path,monkeypatch):
    run,fast,link,archive=episode(tmp_path)
    copy=n.shutil.copytree
    def corrupt(source,destination,*args,**kwargs):
        result=copy(source,destination,*args,**kwargs)
        if Path(source).parent==fast:
            (Path(destination)/'recording/episode.h5').write_bytes(b'corrupt')
        return result
    monkeypatch.setattr(n.shutil,'copytree',corrupt)
    with pytest.raises(ValueError,match='checksum'):n.archive_one(link,fast,archive)
    assert link.resolve()==fast/link.name
    assert (link/'recording/episode.h5').read_bytes()==b'original RGB and joint data'


def test_live_process_cannot_be_archived(tmp_path):
    import os
    run,fast,link,archive=episode(tmp_path)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    (link/'process.json').write_text(json.dumps({'pid':os.getpid(),'proc_start_ticks':ticks}))
    with pytest.raises(ValueError,match='active worker'):n.archive_one(link,fast,archive)
    assert link.resolve()==fast/link.name


def test_recovery_of_verified_destination_before_link_switch(tmp_path):
    run,fast,link,archive=episode(tmp_path)
    destination=archive/'success'/link.name
    destination.parent.mkdir(parents=True);n.shutil.copytree(link,destination)
    assert n.archive_one(link,fast,archive)
    assert link.resolve()==destination


def test_admission_reserves_future_growth_and_system_disk_floor(tmp_path,monkeypatch):
    run,fast,link,archive=episode(tmp_path)
    monkeypatch.setattr(n,'size',lambda p:4*n.GIB)
    monkeypatch.setattr(n.shutil,'disk_usage',lambda p:SimpleNamespace(free=155*n.GIB))
    config={'episode_reserve_gib':16,'fast_min_free_gib':128}
    state=n.capacity(fast,[{'folder':link}],config)
    assert state['reserved_remaining_gib']==12 and not state['admit']
    monkeypatch.setattr(n.shutil,'disk_usage',lambda p:SimpleNamespace(free=156*n.GIB))
    assert n.capacity(fast,[{'folder':link}],config)['admit']


def test_unrelated_symlink_is_never_archived(tmp_path):
    run,fast,link,archive=episode(tmp_path)
    assert not n.archive_one(link,tmp_path/'other',archive)
    assert link.resolve()==fast/link.name
