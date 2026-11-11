"""Finish read-only audits and review exports after the owned recording closes."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,time
root=Path(__file__).resolve().parent
source=Path('outputs/rim_bowl_clearance_rgb_v2/seed1/episode/episode.h5')
trace=Path('outputs/rim_bowl_clearance_rgb_v2/seed1/trace')
audit=source.with_name('audit.json')
pid=json.loads((root/'bowl_clearance_lzf_process.json').read_text())['pid']
for _ in range(480):
    if audit.exists() and (trace/'trace.json').exists():break
    if not Path(f'/proc/{pid}').exists():raise RuntimeError('Recording ended without final audit and trace metadata')
    time.sleep(5)
else:raise TimeoutError('Recording wait exceeded 40 minutes')
a=json.loads(audit.read_text())
assert a['passed'] and a['training_ready'],a.get('violations')
identity=root/'bowl_clearance_lzf_identity.json';d=json.loads(identity.read_text())
def changed_files():
    return [p for p,h in d['files'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed_files(),changed_files()
python='.venv-mujoco/bin/python'
baseline='outputs/rim_bowl_clearance_raw_v1/seed1/episode/episode.h5'
jobs={
    'native':[python,'scripts/analyze_contact_trace.py',str(trace),'--recording',str(source),'--baseline',baseline,'--identity',str(identity),'--output',str(root/'bowl_clearance_native_audit.json')],
    'wrench':[python,'scripts/audit_payload_wrench.py',str(trace),'--recording',str(source),'--identity',str(identity),'--output',str(root/'bowl_clearance_wrench')],
    'preview':[python,'scripts/preview_sensor_rgb.py',str(source),'--policy','configs/collection/quality_policy_tcp.yaml','--output',str(root/'bowl_clearance_sensor_preview')],
    'full_video':[python,'scripts/export_full_sensor_rgb.py',str(source),'--policy','configs/collection/quality_policy_tcp.yaml','--output',str(root/'bowl_clearance_complete_video')],
}
def run(item):
    name,cmd=item
    with (root/f'bowl_clearance_{name}_job.log').open('x') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src','MUJOCO_GL':'egl'})
    print(name,r.returncode,flush=True)
    return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=dict(pool.map(run,jobs.items()))
assert not any(results.values()),results
changed=changed_files();assert not changed,changed
(root/'bowl_clearance_frozen_after.json').write_text(json.dumps({'identity':str(identity),'checked_files':len(d['files']),'changed':changed,'all_closed':True,'review_jobs':results},indent=2)+'\n')
print('Complete RGB, native force consistency, full video and frozen-source checks finished.',flush=True)
