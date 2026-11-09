"""Bind the completed RGB regression, independent consumers and performance scope."""
import json
from pathlib import Path
import shutil
from piper_sim.collection_jobs import atomic_json, sha256

base = Path(__file__).resolve().parent
read = lambda name: json.loads((base/name).read_text())
output = base/'tcp_clearance_rgb_summary_v1.json'
assert not output.exists()
post = read('tcp_clearance_rgb_postcheck_v1.json'); assert post['status'] == 'all_checks_passed'
v = read('tcp_clearance_rgb_validation_v1.json'); assert v['status'] == 'passed_full_rgb_reader_regression'
consumer = read('tcp_clearance_rgb_consumer_v1.json'); assert consumer['status'] == 'consumer_check_passed'
video = read('tcp_clearance_rgb_video_audit_v1.json'); assert video['status'] == 'all_video_frames_and_pts_checked'
fk = read('tcp_clearance_rgb_camera_fk_v1.json'); assert all(r['numerical_fk_check_passed'] for r in fk['results'])
profile = read('tcp_clearance_rgb_evidence_v1/profile.json'); assert profile['returncode'] == 0
audit = read('tcp_clearance_rgb_evidence_v1/audit.json'); assert audit['passed'] and audit['training_ready']
rollout = read('tcp_clearance_rgb_evidence_v1/report.json'); assert rollout['completed'] and rollout['training_ready']
assert v['source_sha256'] == post['source_sha256'] == video['source_sha256'] == audit['source_sha256']
assert v['frames'] == consumer['frames'] == audit['checks']['streams']['frames'] == 8771
for step in post['steps']:
 assert step['returncode'] == 0 and sha256(step['report']) == step['report_sha256']
 assert sha256(step['log']) == step['log_sha256']
 dest = base/'tcp_clearance_rgb_evidence_v1'/Path(step['log']).name
 assert not dest.exists(); shutil.copyfile(step['log'], dest)
for camera, value in video['cameras'].items():
 assert value['frames'] == v['frames'] and value['fps'] == 30 and value['max_pts_error_s'] < 1e-6
source = Path(v['source']); assert sha256(source) == v['source_sha256']
exports = {str(p.relative_to(source.parent/'lerobot_v2')): sha256(p)
           for p in (source.parent/'lerobot_v2').rglob('*') if p.is_file()}
assert exports == post['export_checksums']
inputs = read('tcp_clearance_rgb_inputs_v1.json')
assert all(sha256(p) == d for p, d in inputs['files'].items())
wall = profile['wall_seconds']; frames = v['frames']; resources = profile['resources']
reports = {str(p.relative_to(base)): sha256(p) for p in sorted(base.glob('tcp_clearance_rgb*')) if p.is_file()}
reports.update({str(p.relative_to(base)): sha256(p) for p in sorted((base/'tcp_clearance_rgb_evidence_v1').rglob('*')) if p.is_file()})
result = dict(status='passed_known_full_rgb_regression', collection_training_ready=True, durable_release=False,
 source=str(source), source_sha256=v['source_sha256'], acceptance_policy=audit['acceptance_policy'],
 complete_target_count=len(rollout['results']), targets_succeeded=sum(bool(r['success']) for r in rollout['results']),
 frames=frames, unique_raw_frames_per_camera=v['unique_frames_per_camera'], fps=30,
 recorded_simulated_duration_s=frames/30, full_simulation_clock_end_s=rollout['sim_seconds'],
 raw_rgb_images_compared=v['raw_rgb_images_compared'], exact_physics_rows=v['exact_physics_rows'],
 measured_motion=audit['checks']['task']['motion']['metrics'], tcp_motion=audit['checks']['tcp_motion'],
 all_frozen_inputs_unchanged=len(inputs['files']), source_and_export_unchanged=True,
 private_fields_absent_from_public_reader=True, official_reader_sample_indices=[r['index'] for r in consumer['checks']],
 video={k:{field:x[field] for field in ('frames','fps','max_pts_error_s','psnr_db','mean_absolute_pixel_error_0_to_255','max_channel_absolute_error_0_to_255')}
        for k,x in video['cameras'].items()}, camera_fk=fk['results'][0]['cameras'],
 source_bytes=v['source_bytes'], export_bytes=v['export_bytes'], raw_transitions_per_decimal_GB=v['raw_transitions_per_decimal_GB'],
 raw_and_export_transitions_per_decimal_GB=frames/((v['source_bytes']+v['export_bytes'])/1e9),
 performance=dict(scope='One known regression, instrumented shared host, tmpfs; not durable-disk or population throughput.',
   acquisition_audit_export_wall_s=wall, recorded_host_span_s=v['clocks']['host_observation_span_s'],
   frames_per_recorded_host_second=frames/v['clocks']['host_observation_span_s'],
   frames_per_pipeline_wall_second=frames/wall, single_known_episode_rate_per_hour=3600/wall,
   independent_postcheck_wall_s=post['wall_seconds'], postchecks_are_additional=True,
   total_pipeline_plus_postchecks_s=wall+post['wall_seconds'],
   stage_exclusive_seconds={k:r['exclusive_s'] for k,r in profile['stages'].items()},
   stage_inclusive_seconds={k:r['inclusive_s'] for k,r in profile['stages'].items()},
   unattributed_wall_seconds=profile['unattributed_wall_seconds'], resources=resources,
   mean_cpu_percent_one_core=profile['mean_cpu_percent_one_core']),
 evidence_hashes=reports, storage=dict(root_free_bytes_at_summary=shutil.disk_usage(base).free,
   tmpfs_free_bytes_at_summary=shutil.disk_usage(source).free, raw_on_volatile_tmpfs=True,
   note='Initial 5 GiB root reserve was a planning budget; available bytes fluctuate on this shared host. No raw files were deleted. Raw/export still require durable archival.'),
 limitations=['One known repaired scene; not an unseen-layout reliability estimate or a new 2x speedup.',
 'Per-target finger material switching remains; real camera, contact and robot response are uncalibrated.',
 'Sparse visual review finds hard shadows/simplified materials and low-information idle left-wrist views.',
 'Full decoded video diagnostics quantify lossy encoding, not perceptual or real-world fidelity.',
 'Independent postchecks are additional to normal collection/audit/export profiling; some small visual sampling overlapped postchecks.'])
atomic_json(output, result)
print(json.dumps({k:result[k] for k in ('status','frames','targets_succeeded','source_bytes','export_bytes')},indent=2))
