#!/usr/bin/env python3
"""Run the privileged sorting demo; assisted output is explicitly labelled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene', default='configs/tasks/clean_table.yaml')
    parser.add_argument('--output', type=Path, default=Path('outputs/clean_table_demo')/time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--grasp-mode', choices=['assisted','physical'], default='assisted')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--randomize', action='store_true', help='Experimental; not a validated random-scene expert')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--record', action='store_true', help='Record 30 Hz transitions and privileged state to HDF5')
    parser.add_argument('--rgb', action='store_true', help='Also capture all three camera streams (larger HDF5)')
    parser.add_argument('--lerobot', action='store_true', help='Export the completed HDF5 episode to LeRobot v2.1')
    parser.add_argument('--objects', nargs='+', help='Diagnostic subset; full demo sorts eight objects')
    args = parser.parse_args()
    if args.lerobot and not args.record:
        parser.error('--lerobot requires --record')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    args.output.mkdir(parents=True)
    if not args.viewer:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
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
                obs.extras['teacher'] = {'name': 'clean_table_privileged_v1', 'grasp_mode': self.teacher.mode,
                    'phase': self.teacher.phase, 'object': self.teacher.current,
                    'active_assist': self.teacher.assist.active[1:] if self.teacher.assist and self.teacher.assist.active else None}
            return obs
        def metadata(self):
            return {**super().metadata(), 'teacher': self.teacher.metadata()}

    cfg = load(args.config);cfg['task_scene'] = args.task_scene
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
        session.reset(args.seed, args.randomize)
        if args.video:
            encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                '-f','rawvideo','-pix_fmt','rgb24','-s','1000x800','-r','15','-i','pipe:0',
                '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(args.output/'demo.mp4')], stdin=subprocess.PIPE)
        def frame_hook(t, obs):
            nonlocal frame_count
            frame_count += 1
            if encoder is not None and frame_count % 2 == 0:
                im = Image.fromarray(backend._render(camera,1000,800))
                draw = ImageDraw.Draw(im)
                draw.rectangle((0,0,1000,66),fill=(15,20,28))
                draw.text((18,10),f'PRIVILEGED SORTING DEMO | {args.grasp_mode.upper()} GRASP | t={obs.sim_time:.1f}s',fill='white',font=font)
                count = obs.extras['task']['metrics']['summary']['sorted_count']
                draw.text((18,37),f'{t.current or "settling"}: {t.phase} | contained: {count}/8 | placemat stays on table',fill=(255,210,100),font=font)
                encoder.stdin.write(im.tobytes())
        teacher = CleanTableTeacher(session, grasp_mode=args.grasp_mode, frame_hook=frame_hook)
        session.teacher = teacher
        if args.record:
            session.start_recording('episode', '纸巾放左盒；全部餐具放右篮；餐垫留桌面')
        summary = teacher.run(args.objects)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr, flush=True)
    finally:
        if teacher and teacher.assist:
            teacher.assist.release()
        if backend:
            obs = backend.read()
            if summary is None:
                summary = obs.extras['task']['metrics']['summary']
            if args.video:
                Image.fromarray(backend._render(camera,1000,800)).save(args.output/'final.png')
            report = {'completed': error is None, 'error': error, 'summary': summary,
                      'teacher': teacher.metadata() if teacher else None,
                      'results': teacher.results if teacher else [],
                      'events': teacher.events if teacher else [],
                      'final_objects': obs.extras['objects'], 'final_metrics': obs.extras['task']['metrics'],
                      'seed': args.seed, 'randomized': args.randomize, 'requested_objects':args.objects,
                      'sim_seconds': backend.sim_time}
            (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        if session and session.writer:
            recording = session.stop_recording(success=error is None and summary['success'])
        if encoder:
            encoder.stdin.close()
            if encoder.wait(timeout=30):
                error = error or 'Video encoding failed'
        if session:
            session.close()
        elif backend:
            backend.close()
    if args.lerobot and recording and error is None:
        from piper_sim.lerobot_export import export_lerobot_v2
        export_lerobot_v2([recording['path']], args.output/'lerobot_v2')
    print(json.dumps({'output':str(args.output.resolve()), 'error':error, 'summary':summary},ensure_ascii=False),flush=True)
    return 2 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
