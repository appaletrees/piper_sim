"""Bounded NVMe episode staging and verified, atomic background archival.

Canonical paths remain in the original run. Only closed terminal episode links
are archived; live recordings are never copied, moved or opened by the archiver.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

GIB = 1024**3


def atomic_json(path, value):
    tmp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('w') as f:
        json.dump(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def size(folder):
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())


def capacity(fast, active, config):
    """Reserve remaining full-episode capacity, not just current file sizes."""
    reserve = config['episode_reserve_gib']*GIB
    remaining = sum(max(0, reserve-size(j['folder'])) for j in active
                    if j['folder'].is_symlink() and j['folder'].resolve().is_relative_to(fast))
    free = shutil.disk_usage(fast).free
    required = remaining+reserve+config['fast_min_free_gib']*GIB
    return {'admit': free >= required, 'free_gib': free/GIB,
            'reserved_remaining_gib': remaining/GIB, 'next_required_free_gib': required/GIB}


def create_folder(run, fast, name):
    if not name.startswith('episode_') or '/' in name:
        raise ValueError('Invalid episode name')
    target = fast/name
    target.mkdir()
    link = run/'running'/name
    try:
        link.symlink_to(target, target_is_directory=True)
    except BaseException:
        target.rmdir(); raise
    return link


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):h.update(block)
    return h.hexdigest()


def tree_manifest(folder):
    result = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():raise ValueError('Nested links are forbidden in an episode')
        if p.is_file():result[p.relative_to(folder).as_posix()] = {'size':p.stat().st_size,'sha256':digest(p)}
    return result


def archive_one(link, fast, archive):
    if not link.is_symlink():return False
    source = link.resolve(strict=True)
    if source.parent != fast or source.name != link.name:
        return False  # An already archived link or unrelated link is untouched.
    result = json.loads((source/'result.json').read_text())
    if result.get('category') != link.parent.name or result.get('directory') != str(link):
        raise ValueError('Terminal episode identity mismatch')
    marker = json.loads((source/'process.json').read_text())
    proc = Path('/proc')/str(marker['pid'])
    try:
        stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
        if stat[19] == marker['proc_start_ticks'] and stat[0] != 'Z':
            raise ValueError('Cannot archive an active worker')
    except FileNotFoundError:pass
    destination = archive/link.parent.name/link.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.partial')
    before = tree_manifest(source)
    if destination.exists():
        if tree_manifest(destination) != before:raise ValueError('Conflicting archive destination')
    else:
        if partial.exists():shutil.rmtree(partial)
        shutil.copytree(source, partial)
        for p in partial.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        if tree_manifest(partial) != before:raise ValueError('Archive checksum mismatch')
        for p in sorted((p for p in partial.rglob('*') if p.is_dir()),reverse=True):sync_directory(p)
        sync_directory(partial)
        partial.rename(destination)
        sync_directory(destination.parent)
    # Receipt is durable before the canonical symlink switches and staging is
    # reclaimed. No recording or audit content is modified.
    receipts = archive/'receipts';receipts.mkdir(exist_ok=True)
    atomic_json(receipts/(link.name+'.json'), {'source':str(source),'destination':str(destination),
        'canonical':str(link),'files':before,'verified_unix':time.time()})
    tmp = link.with_name('.'+link.name+'.archive-'+uuid.uuid4().hex)
    tmp.symlink_to(destination, target_is_directory=True)
    if link.resolve() != source:raise ValueError('Canonical link changed during archive')
    locks=link.parent.parent/'storage_locks';locks.mkdir(exist_ok=True)
    with (locks/(link.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.replace(tmp,link)
        sync_directory(link.parent)
        shutil.rmtree(source)
    return True


def archive_loop(run, fast, archive):
    archive.mkdir(parents=True, exist_ok=True)
    with (archive/'.archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                candidates = [p for category in ('success','failed') for p in (run/category).iterdir()
                              if p.is_symlink() and p.resolve().parent == fast]
                for link in sorted(candidates, key=lambda p:p.name):
                    atomic_json(archive/'status.json',{'status':'copying','canonical':str(link),'updated_unix':time.time()})
                    archive_one(link,fast,archive)
                atomic_json(archive/'status.json',{'status':'idle','updated_unix':time.time()})
                status=json.loads((run/'status.json').read_text())
                if status.get('status')=='completed' and not candidates:return
            except Exception as exc:
                atomic_json(archive/'status.json',{'status':'error','error':type(exc).__name__+': '+str(exc),'updated_unix':time.time()})
                print('ARCHIVE_RETRY '+repr(exc),flush=True)
            time.sleep(10)
