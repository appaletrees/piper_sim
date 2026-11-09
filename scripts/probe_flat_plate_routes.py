#!/usr/bin/env python3
"""Query alternative Cartesian routes for the flat plate placement experiment.

Never approves the obstructed original joint path. Each alternate command curve
is checked by the original collision gates and returned explicitly in evidence.
This CLI restores isolated poses only and does not execute any trajectory.
"""
import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import SortFailure
import probe_flat_plate as base


def search_with_routes(teacher, diagnostics):
    original = teacher.check_joint_path
    inside_route = False
    diagnostics['route_probe'] = {'scope': __doc__, 'script_sha256': sha256(__file__),
                                'rise_m': .06, 'curve_source': 'original plan_pose_path and original collision gates'}

    def check(side, start, end, *args, **kwargs):
        nonlocal inside_route
        try:
            return original(side, start, end, *args, **kwargs)
        except SortFailure as rejection:
            if inside_route or not diagnostics.get('attempts'):
                raise
            attempt = diagnostics['attempts'][-1]
            stage = attempt['stage']
            if stage not in ('transfer_path', 'lowering_path') or args or kwargs:
                raise
            routes = attempt.setdefault('alternate_routes', [])
            routes.append({'stage': stage, 'original_joint_path_error': str(rejection)})
            # A route cannot fix a collision at its required endpoint.
            original(side, end, end)
            initial = teacher.q.copy();initial[7:13] = start
            p, rotation = teacher.ik[side].forward(start)
            goal, goal_r = teacher.ik[side].forward(end)
            choices = [('direct_cartesian', [(goal, goal_r)])]
            if stage == 'transfer_path':
                choices.append(('elevated_cartesian', [(p+[0, 0, .06], rotation),
                               (goal+[0, 0, .06], goal_r), (goal, goal_r)]))
            inside_route = True
            try:
                for name, poses in choices:
                    route = {'stage': stage, 'method': name, 'error': None}
                    routes.append(route)
                    try:
                        path = teacher.plan_pose_path(side, poses, start=initial)
                    except SortFailure as exc:
                        route['error'] = str(exc);continue
                    route['commands'] = path.tolist()
                    route['command_samples'] = len(path)
                    route['duration_s'] = len(path)/teacher.fps
                    return
            finally:
                inside_route = False
            raise SortFailure('No checked Cartesian detour for '+stage)

    try:
        teacher.check_joint_path = check
        return original_search(teacher, diagnostics)
    finally:
        teacher.check_joint_path = original


original_search = base.search_flat_plate

if __name__ == '__main__':
    base.search_flat_plate = search_with_routes
    base.main()
