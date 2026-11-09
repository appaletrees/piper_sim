"""Privileged geometric placement search; live state is never modified here.

Vertical support estimates rank candidates, not certify dynamic stability. All
selected motions must still pass the teacher's original robot/payload checks.
"""
import numpy as np


def validate_profile(value):
    if value is None: return None
    keys = {'kinds', 'xy_radius_m', 'xy_step_m', 'wall_margin_m', 'support_gap_m',
            'extra_release_heights_m', 'max_candidates'}
    optional = {'slender_surface_spacing_m'}
    if not isinstance(value, dict) or not keys <= set(value) or set(value)-keys-optional:
        raise ValueError('placement_search requires exactly '+', '.join(sorted(keys)))
    if not isinstance(value['kinds'], list) or not value['kinds'] or any(
            k not in ('chopstick', 'side_dish') for k in value['kinds']):
        raise ValueError('Supported placement search kinds: chopstick, side_dish')
    for key in ('xy_radius_m', 'xy_step_m', 'wall_margin_m', 'support_gap_m'):
        number = value[key]
        if isinstance(number, (str, bool)) or not np.isscalar(number) or not np.isfinite(number) or number <= 0:
            raise ValueError('Positive finite '+key+' required')
    if value['xy_radius_m']/value['xy_step_m'] > 4:
        raise ValueError('Placement grid radius may span at most four steps')
    heights = np.asarray(value['extra_release_heights_m'], float)
    if (heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all()
            or heights[0] != 0 or np.any(np.diff(heights) <= 0)):
        raise ValueError('Release height increments must start at zero and strictly increase')
    if type(value['max_candidates']) is not int or not 1 <= value['max_candidates'] <= 256:
        raise ValueError('Placement max_candidates must be in 1..256')
    if 'slender_surface_spacing_m' in value:
        spacing = value['slender_surface_spacing_m']
        if isinstance(spacing, (str, bool)) or not np.isscalar(spacing) or not np.isfinite(spacing) or not .001 <= spacing <= .01:
            raise ValueError('Slender collision surface spacing must be in .001..01 m')
    return dict(value)


class VerticalSupport:
    """Upper envelope of the actual transformed collision triangles, in metres."""
    def __init__(self, triangles):
        triangles = np.asarray(triangles, float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
            raise ValueError('Finite triangles required')
        a, u, v = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        det = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
        good = abs(det) > 1e-14
        self.a, self.u, self.v, self.det = a[good], u[good], v[good], det[good]
        vertices = triangles[good]
        # Match the barycentric tolerance below, including points just outside
        # a triangle edge; add a conservative floating-point rounding allowance.
        # s,t >= -eps and s+t <= 1+eps permit s or t up to 1+2eps.
        pad = 2e-10*(abs(self.u[:, :2])+abs(self.v[:, :2]))
        pad += 32*np.finfo(float).eps*(1+np.max(abs(vertices[:, :, :2]), axis=1))
        self.lower = vertices[:, :, :2].min(1)-pad
        self.upper = vertices[:, :, :2].max(1)+pad

    def height(self, xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Finite XY points required')
        result = np.full(len(xy), -np.inf)
        if not len(xy): return result
        selected = np.all(self.upper >= xy.min(0), axis=1) & np.all(self.lower <= xy.max(0), axis=1)
        a, u, v, det = (values[selected] for values in (self.a, self.u, self.v, self.det))
        for start in range(0, len(xy), 64):
            delta = xy[start:start+64, None]-a[None, :, :2]
            s = (delta[:, :, 0]*v[:, 1]-delta[:, :, 1]*v[:, 0])/det
            t = (u[:, 0]*delta[:, :, 1]-u[:, 1]*delta[:, :, 0])/det
            inside = (s >= -1e-10) & (t >= -1e-10) & (s+t <= 1+1e-10)
            height = a[:, 2]+s*u[:, 2]+t*v[:, 2]
            if height.shape[1]: result[start:start+64] = np.max(np.where(inside, height, -np.inf), axis=1)
        return result


def collision_surface_points(parts, spacing):
    """Subdivide collision faces along their longest edge, without changing them.

    Final triangle edges are at most spacing metres. This bounds sample gaps;
    it is still sampled support estimation, not an exact collision certificate.
    """
    if not np.isfinite(spacing) or not .001 <= spacing <= .01:
        raise ValueError('Collision sample spacing must be in .001..01 m')
    surfaces = [np.asarray(p['points'], float)[np.asarray(p['faces'])]
                for p in parts if p.get('collision')]
    if not surfaces: raise ValueError('No payload collision surfaces')
    pending = np.concatenate(surfaces)
    if not np.isfinite(pending).all(): raise ValueError('Invalid payload collision surfaces')
    finished = []
    for _ in range(24):
        edges = np.roll(pending, -1, axis=1)-pending
        lengths = np.linalg.norm(edges, axis=2)
        done = lengths.max(1) <= spacing
        finished.append(pending[done].reshape(-1, 3))
        pending = pending[~done]
        if not len(pending): return np.unique(np.concatenate(finished), axis=0)
        longest = lengths[~done].argmax(1)
        row = np.arange(len(pending))
        a, b, c = (pending[row, (longest+i)%3] for i in range(3))
        midpoint = (a+b)/2
        pending = np.concatenate((np.stack((a, midpoint, c), axis=1), np.stack((midpoint, b, c), axis=1)))
        if len(pending) > 200000: raise ValueError('Collision sampling budget exceeded')
    raise ValueError('Collision surface subdivision did not converge')


def support_surface(teacher, target, held):
    """Read current collision geometry, including already placed task objects."""
    task = teacher.b.task
    triangles = []
    names = [n for n in task.active_names if n not in held]
    if target not in names: names.append(target)
    for name in names:
        body = teacher.b.data.body('task/'+name)
        rotation = body.xmat.reshape(3, 3)
        for part in task.parts[name]:
            if not part.get('collision'): continue
            points = np.asarray(part['points'])@rotation.T+body.xpos
            triangles.append(points[np.asarray(part['faces'])])
    return VerticalSupport(np.concatenate(triangles)), names


def diverse_candidates(candidates, nominal, tcp, limit):
    """Keep nominal, reachable-side and margin choices for every yaw tier.

    A scalar margin sort alone can spend the entire IK budget on similar poses.
    The configured slot is a semantic prior, not an immutable release target.
    """
    groups = {}
    for i, row in enumerate(candidates): groups.setdefault(row[6], []).append((i, row))
    selected, used = [], set()
    scores = (lambda c: np.linalg.norm(c[1]-nominal),
              lambda c: (abs(c[1][1]-nominal[1]), np.linalg.norm(c[1]-nominal), np.linalg.norm(c[3][:2]-tcp[:2])),
              lambda c: c[0],
              lambda c: np.linalg.norm(c[3][:2]-tcp[:2]))
    for score in scores:
        for group in groups.values():
            remaining = [(i, c) for i, c in group if i not in used]
            if not remaining: continue
            i, row = min(remaining, key=lambda item: score(item[1]))
            selected.append(row);used.add(i)
            if len(selected) == limit: return selected
    for i, row in sorted(enumerate(candidates), key=lambda item:item[1][0]):
        if i not in used: selected.append(row)
        if len(selected) == limit: break
    return selected


def search(teacher, side, name, target, profile):
    from .clean_table import SortFailure
    tcp, hand_r = teacher.tcp(side)
    obj, basket = (teacher.b.data.body('task/'+n) for n in (name, target))
    offset, relative_r = hand_r.T@(obj.xpos-tcp), hand_r.T@obj.xmat.reshape(3, 3)
    bin_p, bin_r = basket.xpos.copy(), basket.xmat.reshape(3, 3).copy()
    spec = teacher.b.task.specs[target]
    half = np.asarray(spec['size_m'][:2])/2-spec['wall_m']
    rim = bin_p[2]+spec['size_m'][2]
    nominal = np.asarray(teacher.b.task.config['drop_slots_m'].get(name, [0., 0.]))
    points = teacher.b.task.surface_points[name]
    slender = teacher.b.task.specs[name]['kind']=='chopstick'
    spacing = profile.get('slender_surface_spacing_m') if slender else None
    if spacing is not None:
        points = collision_surface_points(teacher.b.task.parts[name], spacing)
    support, obstacles = support_surface(teacher, target, teacher.carried_objects())
    grid = np.arange(-int(profile['xy_radius_m']/profile['xy_step_m']),
                     int(profile['xy_radius_m']/profile['xy_step_m'])+1)*profile['xy_step_m']
    slots = [nominal+[x, y] for x in grid for y in grid]
    yaws = (0, 15, -15, 30, -30, 45, -45, 90, -90, 180) if teacher.b.task.specs[name]['kind']=='chopstick' else (0, 90, 180, -90)
    checked = 0
    failures = []
    for pitch in (0, 15, -15, 30, -30, 45, -45, 60, -60):
        candidates = []
        for yaw in yaws:
            a, t = np.deg2rad([yaw, pitch])
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
            object_r = bin_r@rz@ry
            rotated = points@object_r.T
            desired_r = object_r@relative_r.T
            for slot in slots:
                local = rotated@bin_r+np.r_[slot, 0.]
                margin = float(np.min(half-abs(local[:, :2]).max(0)))
                if slender:
                    # A tilted stick rotates while settling. Require room for
                    # its full horizontal length as well as the release pose;
                    # a narrow projected tilted footprint alone is insufficient.
                    level = points@rz.T+np.r_[slot, 0.]
                    margin = min(margin, float(np.min(half-abs(level[:, :2]).max(0))))
                if margin < profile['wall_margin_m']: continue
                goal = bin_p+bin_r@np.r_[slot, 0.]
                # Rank by wall clearance, then nominal-slot displacement/yaw.
                score = -margin+0.2*np.linalg.norm(slot-nominal)+abs(yaw)*.00001
                candidates.append((score, slot, margin, goal, desired_r, rotated, yaw))
        # Bound each orientation tier separately so unreachable flat poses do
        # not consume the entire search budget before tilted poses are tried.
        ranked = diverse_candidates(candidates, nominal, tcp, max(1, profile['max_candidates']//9))
        for score, slot, margin, goal, desired_r, rotated, yaw in ranked:
            if checked >= profile['max_candidates']: break
            checked += 1
            cloud = rotated+goal
            # Boundary/interior vertices both matter for concave objects. This
            # estimate does not replace the exact compiled collision gate below.
            surface = support.height(cloud[:, :2])
            finite = np.isfinite(surface)
            if not finite.any(): continue
            landing_shift = float(np.max(surface[finite]-cloud[finite, 2]))
            landing = cloud+np.array([0., 0., landing_shift])
            if not slender and ((landing-bin_p)@bin_r)[:, 2].max() > spec['size_m'][2]:
                continue  # Predicted support already holds part above the rim.
            high_obj = goal+np.array([0., 0., rim+.035-cloud[:, 2].min()])
            high = high_obj-desired_r@offset
            try:
                q_high = teacher.solve(side, high, desired_r)
                teacher.check_joint_path(side, teacher.q[(0 if side=='left' else 7):][:6], q_high)
            except SortFailure as error:
                failures.append(str(error));continue
            for extra in profile['extra_release_heights_m']:
                release_shift = landing_shift+profile['support_gap_m']+extra
                low_obj = goal+np.array([0., 0., release_shift])
                if low_obj[2] >= high_obj[2]: continue
                low = low_obj-desired_r@offset
                try:
                    q_low = teacher.solve(side, low, desired_r, q_high)
                    teacher.check_joint_path(side, q_high, q_low)
                    # The lower release can place fingertips inside the basket.
                    # Check the actual release aperture sweep on scratch data,
                    # with sub-millimetre samples, before committing the move.
                    aperture = float(teacher.b.data.qpos[teacher.b.indices[side]['qpos'][6:]].sum())
                    release = .04 if slender else .07  # same existing per-kind release policy
                    widths = np.linspace(aperture, release, max(2, int(np.ceil(abs(release-aperture)/.002))+1))
                    for a, b in zip(widths[:-1], widths[1:]):
                        teacher.check_joint_path(side, q_low, q_low, gripper=(a, b))
                except SortFailure as error:
                    failures.append(str(error));continue
                teacher.events.append({'phase': 'placement_search_selected', 'object': name,
                    'sim_time': teacher.b.sim_time, 'slot_xy_m': slot.tolist(),
                    'object_pitch_deg': pitch, 'object_yaw_deg': yaw, 'wall_margin_m': margin,
                    'support_gap_m': profile['support_gap_m']+extra, 'checked_pose_candidates': checked,
                    'opening_sweep_checked_m': [aperture, release],
                    'rejected_geometry_or_ik': len(failures), 'support_obstacles': obstacles,
                    'support_method': ('subdivided payload collision faces' if spacing is not None else 'visible payload vertices')
                                      +' against vertical envelope of current task collision triangles',
                    'payload_sample_count': len(points), 'payload_surface_spacing_m': spacing,
                    'prediction_note': 'geometric first contact, not dynamic stability proof'})
                return score, high, low, desired_r, q_high, q_low
        if checked >= profile['max_candidates']: break
    raise SortFailure(f'No supported container placement for {name}; checked={checked}; '
                      f'last_failure={failures[-1] if failures else "geometric fit/support"}')
