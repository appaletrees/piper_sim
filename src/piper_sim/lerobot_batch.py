"""CPU-only, resumable conversion of an accepted success manifest to one v2.1 dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .lerobot_export import (export_lerobot_v2, _sha256, _json, _jsonl,
                            _aggregate, _stats, DATA_PATH, VIDEO_PATH)


def atomic_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    _json(temporary,value); os.replace(temporary,path)


def read_json(path):
    return json.loads(Path(path).read_text())


def manifest_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError('Empty success manifest')
    seen=set(); paths=set()
    for row in rows:
        digest=(row.get('verified') or {}).get('source_sha256','')
        if (row.get('accepted') is not True or row.get('training_ready') is not True or
                not isinstance(digest,str) or len(digest)!=64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Every input must be an independently verified accepted episode')
        source=Path(row['path']).resolve()
        if row['index'] in seen or source in paths: raise ValueError('Duplicate input episode')
        if not source.is_file() or source.suffix!='.h5': raise ValueError('Missing closed source HDF5')
        seen.add(row['index']); paths.add(source)
    return sorted(rows,key=lambda row:row['index'])


def shard_path(work, index):
    return Path(work)/'shards'/f'{index:06d}'


def inventory(root):
    return {str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':_sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def verify_shard(work, index, row):
    root=shard_path(work,index); receipt=Path(work)/'receipts'/f'{index:06d}.json'
    if not receipt.exists(): return False
    record=read_json(receipt)
    if (record['source_sha256']!=row['verified']['source_sha256'] or
            record['source_index']!=row['index'] or inventory(root)!=record['files']):
        raise ValueError(f'Checkpoint integrity mismatch: episode {index}')
    return True


def complete_shard(work, index, row):
    root=shard_path(work,index)
    provenance=read_json(root/'meta/sim_episodes.jsonl')
    if provenance['source_sha256']!=row['verified']['source_sha256']:
        raise ValueError('Exported source does not match frozen accepted manifest')
    info=read_json(root/'meta/info.json')
    if info['fps']!=30 or info['total_episodes']!=1:
        raise ValueError('Expected one complete 30 Hz source episode')
    if info['total_frames']!=row['verified']['frames']:
        raise ValueError('Exported frame count differs from accepted manifest')
    atomic_json(Path(work)/'receipts'/f'{index:06d}.json',
                {'source_index':row['index'],'source_sha256':provenance['source_sha256'],
                 'frames':info['total_frames'],'files':inventory(root),'completed_unix':time.time()})


def _export_one(work, index):
    work=Path(work); job=read_json(work/'job.json')
    if _sha256(work/'success.jsonl')!=job['manifest_sha256'] or _sha256(work/'policy.yaml')!=job['policy_sha256']:
        raise ValueError('Frozen conversion inputs changed')
    rows=sorted((json.loads(line) for line in (work/'success.jsonl').read_text().splitlines() if line.strip()),key=lambda row:row['index']); row=rows[index]
    if verify_shard(work,index,row): return
    root=shard_path(work,index)
    # No receipt means integrity was never committed. Retain the orphan and
    # re-encode from the verified HDF5 rather than blessing unknown artifacts.
    if root.exists():
        orphan=work/'uncommitted';orphan.mkdir(exist_ok=True)
        root.rename(orphan/f'{index:06d}_{time.time_ns()}')
    policy_hash=job['episode_policies'][index];policy=work/'policies'/f'{policy_hash}.yaml'
    if _sha256(policy)!=policy_hash:raise ValueError('Frozen episode policy changed')
    export_lerobot_v2(row['path'],root,require_quality=True,policy=policy,
                     progress=lambda message:print(message,flush=True))
    complete_shard(work,index,row)


def export_one(work, index):
    work=Path(work);locks=work/'episode_locks';locks.mkdir(exist_ok=True)
    with (locks/f'{index:06d}.lock').open('a') as lock:
        # An old child may still finish after its parent was stopped.
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _export_one(work,index)


def assemble(work, output, rows):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    work=Path(work); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    staging=work/'assembled'
    if staging.exists(): shutil.rmtree(staging)  # Only our uncommitted assembly; shards stay intact.
    staging.mkdir(); records=[]; statistics=[]; provenance=[]; tasks=[]; total=0; first=None
    for index,row in enumerate(rows):
        if not verify_shard(work,index,row): raise ValueError('Missing checkpoint')
        shard=shard_path(work,index); info=read_json(shard/'meta/info.json')
        if first is None: first=info
        if any(info[k]!=first[k] for k in ['features','fps','robot_type','codebase_version']):
            raise ValueError('Incompatible episode schemas')
        episode=read_json(shard/'meta/episodes.jsonl'); n=episode['length']; task=episode['tasks'][0]
        if task not in tasks: tasks.append(task)
        chunk=index//1000
        table=pq.read_table(shard/DATA_PATH.format(episode_chunk=0,episode_index=0))
        if len(table)!=n: raise ValueError('Checkpoint numeric row count mismatch')
        stats=read_json(shard/'meta/episodes_stats.jsonl')['stats']
        columns={'episode_index':np.full(n,index,dtype=np.int64),
                 'index':np.arange(total,total+n,dtype=np.int64),
                 'task_index':np.full(n,tasks.index(task),dtype=np.int64)}
        for key,values in columns.items():
            table=table.set_column(table.schema.get_field_index(key),key,pa.array(values))
            stats[key]=_stats(values)
        destination=staging/DATA_PATH.format(episode_chunk=chunk,episode_index=index)
        destination.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,destination,compression='snappy')
        for key,feature in info['features'].items():
            if feature['dtype']!='video':continue
            src=shard/VIDEO_PATH.format(episode_chunk=0,episode_index=0,video_key=key)
            dst=staging/VIDEO_PATH.format(episode_chunk=chunk,episode_index=index,video_key=key)
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        extra=f'extras/chunk-{chunk:03d}/episode_{index:06d}.jsonl'
        dst=staging/extra;dst.parent.mkdir(parents=True,exist_ok=True)
        os.link(shard/'extras/chunk-000/episode_000000.jsonl',dst)
        prov=read_json(shard/'meta/sim_episodes.jsonl');prov.update(episode_index=index,frame_extras_path=extra,
             source_manifest_index=row['index'])
        records.append({'episode_index':index,'tasks':[task],'length':n})
        statistics.append(stats);provenance.append(prov);total+=n
    info={**first,'total_episodes':len(rows),'total_frames':total,'total_tasks':len(tasks),
          'total_videos':sum(f['dtype']=='video' for f in first['features'].values())*len(rows),
          'total_chunks':(len(rows)+999)//1000,'splits':{'train':f'0:{len(rows)}'}}
    _json(staging/'meta/info.json',info);_jsonl(staging/'meta/episodes.jsonl',records)
    _jsonl(staging/'meta/tasks.jsonl',({'task_index':i,'task':t} for i,t in enumerate(tasks)))
    _jsonl(staging/'meta/episodes_stats.jsonl',({'episode_index':i,'stats':s} for i,s in enumerate(statistics)))
    _json(staging/'meta/stats.json',_aggregate(statistics));_jsonl(staging/'meta/sim_episodes.jsonl',provenance)
    report={**read_json(shard_path(work,0)/'meta/export.json'),'output':str(output),
            'episodes':len(rows),'frames':total,'videos':info['total_videos'],
            'manifest_sha256':read_json(work/'job.json')['manifest_sha256'],
            'quality_required':True,'acceptance_policy_whitelist':read_json(work/'job.json')['policy_whitelist'],
            'checkpoint_work_directory':str(work),'completed_unix':time.time()}
    _json(staging/'meta/export.json',report)
    staging.rename(output)
    return report


def run(manifest, output, policy, *, workers=2, min_free_gib=128, additional_policies=()):
    output=Path(output).expanduser().resolve();work=output.with_name(output.name+'.conversion')
    if workers<1 or workers>8 or min_free_gib<1:raise ValueError('Invalid CPU worker/storage limits')
    output.parent.mkdir(parents=True,exist_ok=True);work.mkdir(exist_ok=True)
    with (work/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise FileExistsError(output)
        rows=manifest_rows(manifest);raw=Path(manifest).read_bytes();policy_bytes=Path(policy).read_bytes()
        policies={hashlib.sha256(Path(p).read_bytes()).hexdigest():Path(p).read_bytes()
                  for p in [policy,*additional_policies]}
        episode_policies=[]
        for row in rows:
            audit=read_json(Path(row['path']).with_name('audit.json'))
            digest=audit.get('acceptance_policy',{}).get('sha256')
            if digest not in policies:raise ValueError('Episode policy is not in the operator whitelist')
            episode_policies.append(digest)
        code=hashlib.sha256()
        for source in sorted(Path(__file__).parent.rglob('*.py')):
            code.update(str(source.relative_to(Path(__file__).parent)).encode());code.update(source.read_bytes())
        expected={'converter_sha256':code.hexdigest(),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
                  'policy_sha256':hashlib.sha256(policy_bytes).hexdigest(),
                  'policy_whitelist':sorted(policies),'episode_policies':episode_policies,'episodes':len(rows),
                  'output':str(output),'format':'LeRobot Dataset v2.1'}
        job_file=work/'job.json'
        if job_file.exists():
            old=read_json(job_file)
            if any(old[k]!=v for k,v in expected.items()):raise ValueError('Conversion inputs differ from checkpoint job')
            if (work/'success.jsonl').read_bytes()!=raw or (work/'policy.yaml').read_bytes()!=policy_bytes:
                raise ValueError('Frozen conversion inputs changed')
        else:
            (work/'success.jsonl').write_bytes(raw);(work/'policy.yaml').write_bytes(policy_bytes)
            atomic_json(job_file,{**expected,'created_unix':time.time()})
        policy_dir=work/'policies';policy_dir.mkdir(exist_ok=True)
        for digest,raw_policy in policies.items():
            target=policy_dir/f'{digest}.yaml'
            if target.exists() and target.read_bytes()!=raw_policy:raise ValueError('Frozen policy changed')
            if not target.exists():target.write_bytes(raw_policy)
        logs=work/'logs';logs.mkdir(exist_ok=True)
        done=set();active={};failures={};started=time.time();status='running';error=None
        def publish():
            atomic_json(work/'status.json',{'status':status,'pid':os.getpid(),'updated_unix':time.time(),
                'started_unix':started,'total':len(rows),'completed':len(done),'active':sorted(active),
                'failed':failures,'error':error,'workers':workers,'output':str(output),'gpu_used':False,
                'free_gib':shutil.disk_usage(output.parent).free/2**30})
        try:
            for i,row in enumerate(rows):
                if verify_shard(work,i,row):done.add(i)
            pending=iter(i for i in range(len(rows)) if i not in done)
            exhausted=False
            def worker(index):
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                cmd=[sys.executable,'-u','-m','piper_sim.lerobot_batch','--work-dir',str(work),'--episode-index',str(index)]
                with (logs/f'{index:06d}.log').open('a') as log:
                    return subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while active or not exhausted:
                    while len(active)<workers and not exhausted and not failures:
                        if shutil.disk_usage(output.parent).free/2**30<min_free_gib:
                            raise OSError('Conversion stopped below configured free-space reserve; checkpoints retained')
                        i=next(pending,None)
                        if i is None:exhausted=True;break
                        active[i]=pool.submit(worker,i)
                    publish()
                    if not active:break
                    finished,_=wait(active.values(),timeout=5,return_when=FIRST_COMPLETED)
                    for index,future in list(active.items()):
                        if future not in finished:continue
                        code=future.result();del active[index]
                        if code or not verify_shard(work,index,rows[index]):
                            failures[index]={'returncode':code,'log':str(logs/f'{index:06d}.log')}
                        else:done.add(index)
                    if failures:exhausted=True
            if failures:raise RuntimeError('One or more episodes failed; see per-episode logs; nothing silently skipped')
            status='assembling';publish();report=assemble(work,output,rows)
            status='completed';publish();return report
        except BaseException as exc:
            error=type(exc).__name__+': '+str(exc);status='failed';publish();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--policy',type=Path);parser.add_argument('--additional-policy',type=Path,action='append',default=[])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--min-free-gib',type=float,default=128)
    parser.add_argument('--work-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--episode-index',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.work_dir is not None:
        if args.episode_index is None:parser.error('Worker requires an episode index')
        return export_one(args.work_dir,args.episode_index)
    if not all([args.manifest,args.output,args.policy]):parser.error('--manifest --output --policy are required')
    print(json.dumps(run(args.manifest,args.output,args.policy,workers=args.workers,
                         min_free_gib=args.min_free_gib,additional_policies=args.additional_policy),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
