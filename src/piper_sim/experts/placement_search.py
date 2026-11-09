"""Privileged geometric placement search; live state is never modified here.

Vertical support estimates rank candidates, not certify dynamic stability. All
selected motions must still pass the teacher's original robot/payload checks.
"""
import numpy as np


def validate_profile(value):
    if value is None: return None
    keys = {'kinds', 'xy_radius_m', 'xy_step_m', 'wall_margin_m', 'support_gap_m',
            'extra_release_heights_m', 'max_candidates'}
    if not isinstance(value, dict) or set(value) != keys:
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

    def height(self, xy):
        xy = np.asarray(xy, float)
        result = np.full(len(xy), -np.inf)
        for start in range(0, len(xy), 64):
            delta = xy[start:start+64, None]-self.a[None, :, :2]
            s = (delta[:, :, 0]*self.v[:, 1]-delta[:, :, 1]*self.v[:, 0])/self.det
            t = (self.u[:, 0]*delta[:, :, 1]-self.u[:, 1]*delta[:, :, 0])/self.det
            inside = (s >= -1e-10) & (t >= -1e-10) & (s+t <= 1+1e-10)
            height = self.a[:, 2]+s*self.u[:, 2]+t*self.v[:, 2]
            if height.shape[1]: result[start:start+64] = np.max(np.where(inside, height, -np.inf), axis=1)
        return result


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
                    'support_method': 'object-vertex samples against vertical envelope of current task collision triangles',
                    'prediction_note': 'geometric first contact, not dynamic stability proof'})
                return score, high, low, desired_r, q_high, q_low
        if checked >= profile['max_candidates']: break
    raise SortFailure(f'No supported container placement for {name}; checked={checked}; '
                      f'last_failure={failures[-1] if failures else "geometric fit/support"}')
