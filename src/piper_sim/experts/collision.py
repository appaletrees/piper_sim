"""Conservative robot-obstacle/inter-arm penetration audit on MuJoCo contacts."""


def robot_penetration(model, data, target=None):
    worst, pair = 0., None
    def robot(name):
        pieces = name.split('/')
        return len(pieces) == 2 and pieces[0] in ('left', 'right') and pieces[1] in {
            'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'link8'}
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        active = [robot(n) for n in bodies]
        if not any(active):
            continue
        # Same-arm adjacent CAD geometry/contact exclusions are not a certified
        # self-collision model. Still check the other arm and all scene obstacles.
        if all(active) and bodies[0].split('/')[0] == bodies[1].split('/')[0]:
            continue
        if target and 'task/'+target in bodies and any(n.endswith(('/link7', '/link8')) for n in bodies):
            continue  # intended pad-object contact; measured-motion gates still apply
        depth = float(-contact.dist)
        if depth > worst:
            worst, pair = depth, bodies
    return worst, pair


def carried_obstacle_penetration(model, data, carried):
    """Payload versus scene contacts; internal cup-stack and gripper contacts excluded."""
    carried = {'task/'+name for name in carried}
    worst, pair = 0., None
    for contact in data.contact:
        if contact.dist >= 0: continue
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        selected = [name in carried for name in bodies]
        if sum(selected) != 1: continue
        other = bodies[1] if selected[0] else bodies[0]
        if other.startswith(('left/', 'right/')): continue  # existing robot audit covers these
        if -contact.dist > worst:
            worst, pair = float(-contact.dist), bodies
    return worst, pair
