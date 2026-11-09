#!/usr/bin/env python3
"""Run the privileged sorting demo; assisted output is explicitly labelled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene', default='configs/tasks/clean_table_rigid.yaml')
    parser.add_argument('--output', type=Path, default=Path('outputs/clean_table_demo')/time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--grasp-mode', choices=['assisted','physical'], default='physical')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--randomize', action='store_true', help='Enable reproducible task/domain randomization')
    parser.add_argument('--domain-randomization', help='Episode randomization YAML, e.g. configs/domain_randomization/clean_table.yaml')
    parser.add_argument('--collection-profile', default='configs/collection/clean_table.yaml')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video-speed', type=int, default=1, choices=range(1,17),
                        help='Preview playback speed; does not change physics/control/recording rate')
    parser.add_argument('--record', action='store_true', help='Record 30 Hz transitions and privileged state to HDF5')
    parser.add_argument('--rgb', action='store_true', help='Also capture all three camera streams (larger HDF5)')
    parser.add_argument('--lerobot', action='store_true', help='Export the completed HDF5 episode to LeRobot v2.1')
    parser.add_argument('--objects', nargs='+', help='Diagnostic subset; cup_stack tests nesting and stack transport')
    args = parser.parse_args()
    if args.domain_randomization:
        args.randomize = True
    if args.lerobot and (not args.rgb or args.objects):
        parser.error('--lerobot requires all RGB streams and a complete task (no --objects)')
    if args.lerobot and not args.record:
        parser.error('--lerobot requires --record')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    args.output.mkdir(parents=True)
    if not args.viewer:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from piper_sim.config import load
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.simulation import SimulationSession
    from piper_sim.experts.clean_table import CleanTableTeacher
    from piper_sim.experts.grasp_assist import add_grasp_constraints

    class DemoSession(SimulationSession):
        teacher = None
        def read(self):
            obs = super().read()
            if self.teacher:
                obs.extras['teacher'] = self.teacher.telemetry()
            return obs
        def metadata(self):
            return {**super().metadata(), 'teacher': self.teacher.metadata(), 'provenance': provenance,
                    'collection_initial_target': self.teacher.q.tolist(),
                    'software': {'python': sys.version, 'numpy': np.__version__, 'mujoco': mujoco.__version__}}

    import yaml
    cfg = load(args.config);cfg['task_scene'] = yaml.safe_load(Path(args.task_scene).read_text())
    if args.domain_randomization:
        cfg['task_scene']['domain_randomization'] = args.domain_randomization
    settings = yaml.safe_load(Path(args.collection_profile).read_text())
    import hashlib
    from piper_sim.config import ROOT
    source_files = [p for p in (ROOT/'src/piper_sim').rglob('*.py') if 'rl' not in p.parts]
    source_files += [Path(__file__), Path(args.task_scene), Path(args.collection_profile)]
    if args.domain_randomization:
        source_files.append(Path(args.domain_randomization))
    source_files += list((ROOT/'assets/tasks/clean_table').glob('*'))
    provenance = {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source_files if p.is_file() and p.resolve().is_relative_to(ROOT)}
    backend = None;session = None;teacher = None;encoder = None;error = None;recording = None;summary = None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.40,0,.10];camera.distance = 1.40
    camera.azimuth = 20;camera.elevation = -78
    frame_count = 0
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path,18) if Path(font_path).exists() else ImageFont.load_default()
    try:
        backend = MujocoBackend(cfg, cameras=args.rgb, viewer=args.viewer,
            model_xml_transform=add_grasp_constraints if args.grasp_mode == 'assisted' else None)
        session = DemoSession(backend, record_dir=args.output)
        if backend.task.config.get('variant')=='rigid_stack_v1':
            from piper_sim.experts.rigid_table import RigidTableTeacher
            CleanTableTeacher=RigidTableTeacher
        session.reset(args.seed, args.randomize)
        if args.video:
            encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                '-f','rawvideo','-pix_fmt','rgb24','-s','1000x800','-r','15','-i','pipe:0',
                '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(args.output/'demo.mp4')], stdin=subprocess.PIPE)
        def frame_hook(t, obs):
            nonlocal frame_count
            frame_count += 1
            if args.record and frame_count % 30 == 0:
                reserve = settings.get('dataset', {}).get('min_free_disk_gib', 5.) * 1024**3
                if shutil.disk_usage(args.output).free < reserve:
                    raise OSError('RecordingStopped: free disk space fell below the configured reserve')
            if encoder is not None and frame_count % (2*args.video_speed) == 0:
                im = Image.fromarray(backend._render(camera,1000,800))
                draw = ImageDraw.Draw(im)
                draw.rectangle((0,0,1000,66),fill=(15,20,28))
                draw.text((18,10),f'PRIVILEGED SORTING | {args.grasp_mode.upper()} | {args.video_speed}x PREVIEW | t={obs.sim_time:.1f}s',fill='white',font=font)
                count = obs.extras['task']['metrics']['summary']['sorted_count']
                required = obs.extras['task']['metrics']['summary']['required_count']
                scene_note = 'rigid props; cups stacked first' if backend.task.config.get('variant') == 'rigid_stack_v1' else 'placemat stays on table'
                draw.text((18,37),f'{t.current or "settling"}: {t.phase} | contained: {count}/{required} | {scene_note}',fill=(255,210,100),font=font)
                encoder.stdin.write(im.tobytes())
        teacher = CleanTableTeacher(session, grasp_mode=args.grasp_mode, frame_hook=frame_hook, settings=settings)
        session.teacher = teacher
        teacher.prepare(teacher.cups if args.objects==['cup_stack'] and hasattr(teacher,'cups') else args.objects)
        if args.record:
            reserve = settings.get('dataset', {}).get('min_free_disk_gib', 5.) * 1024**3
            if shutil.disk_usage(args.output).free < reserve:
                raise OSError('RecordingStopped: insufficient free disk space for the configured reserve')
            instruction = ('先在桌面套叠纸杯，再将整摞杯子、笔、充电头和全部餐具放入右篮'
                           if backend.task.config.get('variant') == 'rigid_stack_v1'
                           else '纸巾放左盒；全部餐具放右篮；餐垫留桌面')
            session.start_recording('episode', instruction,
                image_compression=settings.get('dataset', {}).get('image_compression', 'lzf'))
        summary = teacher.run(args.objects)
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr, flush=True)
    finally:
        if teacher and teacher.assist:
            teacher.assist.release()
        if encoder:
            encoder.stdin.close()
            if encoder.wait(timeout=30):
                error = error or 'Video encoding failed'
        if backend:
            obs = backend.read()
            if summary is None:
                summary = obs.extras['task']['metrics']['summary']
            if args.video:
                Image.fromarray(backend._render(camera,1000,800)).save(args.output/'final.png')
            quality = teacher.quality.report() if teacher else {'passed': False, 'violations': ['no teacher']}
            quality['physical_grasps'] = teacher.grasps if teacher else []
            if args.grasp_mode != 'physical':
                quality['violations'].append('assisted_grasp_not_accepted')
                quality['passed'] = False
            quality['cameras'] = teacher.camera_quality if teacher else {}
            if args.rgb:
                for name, stats in quality['cameras'].items():
                    if stats['low_contrast_frames']/max(stats['frames'], 1) > .98:
                        quality['violations'].append('uninformative_camera/'+name)
                if set(quality['cameras']) != {'overhead', 'left_wrist', 'right_wrist'}:
                    quality['violations'].append('missing_camera_frames')
                quality['passed'] = quality['passed'] and not quality['violations']
            if error is None and not quality['passed']:
                error = 'QualityRejected: '+', '.join(quality['violations'])
            accepted = bool(error is None and summary['success'] and not args.objects)
            if session and session.writer:
                session.writer.f.attrs['quality'] = json.dumps(quality)
                session.writer.f.attrs['rollout_accepted'] = accepted
                session.writer.f.attrs['accepted'] = False  # closed-file audit has not run yet
                session.writer.f.attrs['training_ready'] = False
                metadata = json.loads(session.writer.f.attrs['metadata'])
                metadata['collection_outcome'] = {'accepted': False, 'rollout_accepted': accepted, 'error': error,
                    'teacher_events': teacher.events if teacher else [],
                    'object_results': teacher.results if teacher else []}
                session.writer.f.attrs['metadata'] = json.dumps(metadata)
            report = {'accepted': False, 'training_ready': False, 'rollout_accepted': accepted, 'quality': quality,
                      'video_playback_speed':args.video_speed if args.video else None,
                      'robot_collision_peak': teacher.collision_peak if teacher else None,
                      'provenance': provenance,
                      'domain_randomization': backend.task.domain_parameters,
                      'inventory': backend.task.layout['content'],
                      'collection_profile': settings, 'rgb': args.rgb,
                      'completed': error is None, 'error': error, 'summary': summary,
                      'teacher': teacher.metadata() if teacher else None,
                      'results': teacher.results if teacher else [],
                      'events': teacher.events if teacher else [],
                      'final_objects': obs.extras['objects'], 'final_metrics': obs.extras['task']['metrics'],
                      'seed': args.seed, 'randomized': args.randomize, 'requested_objects':args.objects,
                      'sim_seconds': backend.sim_time}
            (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        if session and session.writer:
            recording = session.stop_recording(success=error is None and summary['success'])
        if session:
            session.close()
        elif backend:
            backend.close()
    # Re-open the CLOSED recording: a successful script is not sufficient to
    # publish training data. State-only runs remain motion/task diagnostics.
    if recording and Path(recording['path']).suffix == '.h5':
        from piper_sim.collection_audit import audit_episode, save_audit
        audit = audit_episode(recording['path'], require_rgb=args.rgb)
        save_audit(recording['path'], audit, args.output/'audit.json')
        report['accepted'] = audit['training_ready']
        report['training_ready'] = audit['training_ready']
        report['audit_passed'] = audit['passed']
        report['audit_violations'] = audit['violations']
        report['diagnostic_only'] = bool(args.objects or not args.rgb)
        if not audit['passed'] and not args.objects:
            error = error or 'DatasetRejected: '+', '.join(audit['violations'])
            report['error'] = error
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    if args.lerobot and recording and error is None and report.get('training_ready'):
        from piper_sim.lerobot_export import export_lerobot_v2
        export_lerobot_v2([recording['path']], args.output/'lerobot_v2', require_quality=True)
    print(json.dumps({'output':str(args.output.resolve()), 'error':error, 'summary':summary},ensure_ascii=False),flush=True)
    return 2 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
