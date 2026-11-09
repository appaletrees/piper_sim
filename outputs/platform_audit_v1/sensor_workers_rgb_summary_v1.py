"""Bind worker equivalence, full RGB regression, consumers and scoped timings."""
import json
from pathlib import Path
import shutil

from piper_sim.collection_jobs import atomic_json, sha256

base = Path(__file__).resolve().parent
read = lambda name: json.loads((base/name).read_text())
output = base/'sensor_workers_rgb_summary_v1.json'
assert not output.exists()
post = read('sensor_workers_rgb_postcheck_v1.json')
assert post['status'] == 'all_checks_passed'
v = read('sensor_workers_rgb_validation_v1.json')
assert v['status'] == 'passed_full_rgb_reader_regression'
consumer = read('sensor_workers_rgb_consumer_v1.json')
assert consumer['status'] == 'consumer_check_passed'
video = read('sensor_workers_rgb_video_audit_v1.json')
assert video['status'] == 'all_video_frames_and_pts_checked'
profile = read('sensor_workers_rgb_evidence_v1/profile.json')
assert profile['returncode'] == 0
audit = read('sensor_workers_rgb_evidence_v1/audit.json')
assert audit['passed'] and audit['training_ready'] and not audit['violations']
rollout = read('sensor_workers_rgb_evidence_v1/report.json')
assert rollout['completed'] and rollout['training_ready']
isaac = read('sensor_workers_isaac_check_v2.json')
execution = read('sensor_workers_isaac_execution_v2.json')
assert isaac['status'] == 'passed_actual_isaac_sensor_equivalence'
assert isaac['processed_triplets'] == 52 and isaac['owned_workers_joined']
assert execution['exit_code'] == 0
assert execution['report_sha256'] == sha256(base/'sensor_workers_isaac_check_v2.json')
assert all(sha256(p) == digest for p, digest in isaac['inputs'].items())
gl = read('sensor_workers_gl_check_v2.json')
assert gl['status'] == 'passed_actual_render_equivalence'
assert gl['processed_triplets'] == 64 and gl['different_channel_values'] == 0
assert gl['reference_sha256'] == isaac['reference_sha256'] == sha256(base/'sensor_workers_reference_v1.py')
assert gl['script_sha256'] == sha256(base/'sensor_workers_gl_check_v2.py')
tests = read('sensor_workers_tests_v1.json')
assert tests['status'] == 'final_tests_passed' and tests['runs'][-1]['passed'] == 492
for run in tests['runs']:
    assert sha256(base.parents[1]/run['log']) == run['sha256']
assert v['source_sha256'] == post['source_sha256'] == video['source_sha256'] == audit['source_sha256']
assert v['frames'] == consumer['frames'] == audit['checks']['streams']['frames'] == 8771
for step in post['steps']:
    assert step['returncode'] == 0 and sha256(step['report']) == step['report_sha256']
    assert sha256(step['log']) == step['log_sha256']
    dest = base/'sensor_workers_rgb_evidence_v1'/Path(step['log']).name
    assert not dest.exists()
    shutil.copyfile(step['log'], dest)
for camera, value in video['cameras'].items():
    assert value['frames'] == v['frames'] and value['fps'] == 30 and value['max_pts_error_s'] < 1e-6
source = Path(v['source'])
assert sha256(source) == v['source_sha256']
export_root = source.parent/'lerobot_v2'
exports = {str(p.relative_to(export_root)): sha256(p) for p in export_root.rglob('*') if p.is_file()}
assert exports == post['export_checksums']
inputs = read('sensor_workers_rgb_inputs_v1.json')
assert all(sha256(p) == digest for p, digest in inputs['files'].items())
prior = read('tcp_clearance_rgb_summary_v1.json')
plan = read('sensor_workers_rgb_plan_v1.json')
assert sha256(base/'tcp_clearance_rgb_summary_v1.json') == plan['comparison_summary_sha256']
assert sha256(prior['source']) == plan['comparison_rgb_sha256']
assert sha256(base/'tcp_clearance_rgb_camera_fk_v1.json') == prior['evidence_hashes']['tcp_clearance_rgb_camera_fk_v1.json']
assert v['exact_physics_rows'] == prior['exact_physics_rows'] == 8772
assert 'cameras' in v['exact_physics_extra_fields']
assert audit['acceptance_policy'] == prior['acceptance_policy']
assert audit['checks']['task']['motion']['metrics'] == prior['measured_motion']
baseline = read('tcp_clearance_rgb_evidence_v1/profile.json')
wall = profile['wall_seconds']
old_wall = baseline['wall_seconds']
frames = v['frames']
stage_comparison = {name: {'serial_inclusive_s': old['inclusive_s'],
    'workers3_inclusive_s': profile['stages'][name]['inclusive_s']}
    for name, old in baseline['stages'].items() if name in profile['stages']}
paths = [p for p in sorted(base.glob('sensor_workers*')) if p.is_file()]
paths += [p for p in sorted((base/'sensor_workers_rgb_evidence_v1').rglob('*')) if p.is_file()]
paths += list(sorted(base.glob('pytest_sensor_workers*.log')))
reports = {str(p.relative_to(base)): sha256(p) for p in paths}
assert sum(p.stat().st_size for p in paths) < 2*1024**2
result = dict(
    status='passed_known_full_rgb_worker_regression', collection_training_ready=True, durable_release=False,
    source=str(source), source_sha256=v['source_sha256'], acceptance_policy=audit['acceptance_policy'],
    complete_target_count=len(rollout['results']), targets_succeeded=sum(bool(r['success']) for r in rollout['results']),
    frames=frames, unique_raw_frames_per_camera=v['unique_frames_per_camera'], fps=30,
    recorded_simulated_duration_s=frames/30, full_simulation_clock_end_s=rollout['sim_seconds'],
    raw_rgb_images_compared=v['raw_rgb_images_compared'], exact_physics_and_camera_rows=v['exact_physics_rows'],
    raw_cross_run_pixel_comparison=v['raw_pixel_comparison'],
    raw_pixel_comparison_scope=v['raw_pixel_comparison_scope'],
    measured_motion=audit['checks']['task']['motion']['metrics'], tcp_motion=audit['checks']['tcp_motion'],
    all_frozen_inputs_unchanged=len(inputs['files']), source_and_export_unchanged=True,
    private_fields_absent_from_public_reader=True,
    official_reader_sample_indices=[row['index'] for row in consumer['checks']],
    video={k: {field: row[field] for field in ('frames', 'fps', 'max_pts_error_s', 'psnr_db',
        'mean_absolute_pixel_error_0_to_255', 'max_channel_absolute_error_0_to_255')}
        for k, row in video['cameras'].items()},
    camera_fk_scope='All 8772 camera pose telemetry rows exactly match the serial baseline with a verified complete FK audit; no real-world calibration claim.',
    component_and_engine_equivalence=dict(
        component_report='sensor_workers_probe_v1.json',
        actual_mujoco_report='sensor_workers_gl_check_v2.json',
        actual_isaac_report='sensor_workers_isaac_check_v2.json',
        same_input_outputs_identical=True, mujoco_triplets=64, isaac_triplets=52,
        isaac_scope=execution['scope'], tests_passed=492,
        prior_failures_preserved=['pytest_sensor_workers_v1.log', 'sensor_workers_gl_check_v1_failure.json',
                                 'sensor_workers_isaac_check_v1_unverified.json']),
    source_bytes=v['source_bytes'], export_bytes=v['export_bytes'],
    raw_transitions_per_decimal_GB=v['raw_transitions_per_decimal_GB'],
    raw_and_export_transitions_per_decimal_GB=frames/((v['source_bytes']+v['export_bytes'])/1e9),
    performance=dict(
        scope='One known successful regression, different shared-host time windows, tmpfs; not a paired population or durable-disk benchmark.',
        acquisition_audit_export_wall_s=wall, serial_acquisition_audit_export_wall_s=old_wall,
        single_known_pipeline_throughput_ratio=old_wall/wall, wall_time_reduction_fraction=1-wall/old_wall,
        recorded_host_span_s=v['clocks']['host_observation_span_s'],
        frames_per_recorded_host_second=frames/v['clocks']['host_observation_span_s'],
        frames_per_pipeline_wall_second=frames/wall, single_known_episode_rate_per_hour=3600/wall,
        serial_single_known_episode_rate_per_hour=3600/old_wall,
        independent_postcheck_wall_s=post['wall_seconds'], postchecks_are_additional=True,
        total_pipeline_plus_postchecks_s=wall+post['wall_seconds'],
        postcheck_overlap=execution['overlap'],
        postcheck_comparison_note='New checks also read all old RGB frames; different workload and Isaac overlap prevent a direct postcheck-speed comparison.',
        stage_comparison=stage_comparison,
        stage_exclusive_seconds={k: row['exclusive_s'] for k, row in profile['stages'].items()},
        resources=profile['resources'], mean_cpu_percent_one_core=profile['mean_cpu_percent_one_core']),
    evidence_hashes=reports,
    working_tree_context=dict(
        preexisting_uncommitted_modules={p: digest for p, digest in inputs['files'].items()
            if p.endswith(('/robot_asset.py', '/tasks/isaac.py'))},
        note='Unrelated existing PPO hooks remain uncommitted by this work. Probe uses default arm home and default /World/Task paths; all tested file hashes are recorded. These are tested-workspace results, not a clean-checkout assertion.'),
    storage=dict(root_free_bytes_at_summary=shutil.disk_usage(base).free,
        tmpfs_free_bytes_at_summary=shutil.disk_usage(source).free, raw_on_volatile_tmpfs=True,
        evidence_bytes_before_summary=sum(p.stat().st_size for p in paths),
        note='Only bounded small evidence persisted to root. Raw/export remain volatile and require durable archival. No existing raw files deleted.'),
    limitations=[
        'Known regression only; no unseen-layout reliability estimate or full-pipeline 2x achievement.',
        'Same raw input yields identical CPU processing, but separate GPU runs have a few different channel values (max 2/255); no cross-run pixel identity claim.',
        'Per-target finger material switching remains; real camera, contact and robot response are uncalibrated.',
        'Full Isaac manipulation and cross-engine dynamics equivalence are not established by a short sensor probe.',
        'Sparse prior visual review finds hard shadows/simplified materials and low-information idle left-wrist views.',
        'Full decoded video diagnostics measure lossy encoding, not perceptual or real-world fidelity.',
        'Default remains one sensor worker; candidate config opts into three. Multi-environment contention is unmeasured.'])
atomic_json(output, result)
assert sum(p.stat().st_size for p in paths)+output.stat().st_size < 2*1024**2
print(json.dumps({key: result[key] for key in ('status', 'frames', 'targets_succeeded', 'source_bytes', 'export_bytes')}, indent=2))
