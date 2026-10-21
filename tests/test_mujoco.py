"""Cross-engine contract and source-geometry tests; no GL or hardware needed."""
import time
import shutil
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT, load
from piper_sim.contracts import Command, JOINT_NAMES
from piper_sim.kinematics import forward
from piper_sim.transforms import pose_matrix


@pytest.fixture(scope="module")
def backend():
    if not (ROOT/"assets/mujoco/manifest.json").exists() and not (ROOT/"assets/generated/cad_gripper/manifest.json").exists():
        pytest.skip("Prepare the portable MuJoCo or source CAD assets first")
    backend = MujocoBackend(load())
    yield backend
    backend.close()


@pytest.mark.parametrize("angles", [
    [0, 0, 0, 0, 0, 0, .04], [.3, .8, -.7, -.2, .4, .25, .06],
    [-.5, 1.1, -1.2, .4, -.3, -.6, .015]])
def test_all_link_and_camera_frames_match_source_urdf(backend, angles):
    q = np.array(angles*2)
    backend.set_observation_state(q)
    for side in ("left", "right"):
        base = backend.cfg["robot"][side]
        reference = ROOT/backend.cfg["robot"]["urdf"]
        if not reference.is_file():
            reference = ROOT/"assets/mujoco/kinematics.urdf"
        expected = forward(reference,
            {**dict(zip(JOINT_NAMES, angles[:6])), "joint7": angles[6]/2, "joint8": angles[6]/2},
            pose_matrix(base["position"], base["rotation"]))
        for link, matrix in expected.items():
            body = backend.data.body(f"{side}/{link}")
            np.testing.assert_allclose(body.xpos, matrix[:3, 3], atol=2e-10)
            np.testing.assert_allclose(body.xmat.reshape(3, 3), matrix[:3, :3], atol=2e-10)
        camera = backend.rig["cameras"][f"{side}_wrist"]
        predicted = expected["link6"] @ pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
        cid = backend.model.camera(f"{side}_wrist").id
        np.testing.assert_allclose(backend.data.cam_xpos[cid], predicted[:3, 3], atol=2e-10)
        actual = backend.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        np.testing.assert_allclose(actual, predicted[:3, :3], atol=2e-10)


def test_compiled_meshes_retain_all_source_bounds(backend):
    backend.reset()
    assert len(backend.manifest["cad_meshes"]) == 99
    for record in backend.manifest["cad_meshes"]:
        geom = backend.model.geom(record["name"])
        mesh_id = geom.dataid[0]
        start = backend.model.mesh_vertadr[mesh_id]
        count = backend.model.mesh_vertnum[mesh_id]
        points = backend.model.mesh_vert[start:start+count].astype(float)
        g = backend.data.geom(record["name"])
        world = points @ g.xmat.reshape(3, 3).T + g.xpos
        body = backend.data.body(geom.bodyid[0])
        local = (world-body.xpos) @ body.xmat.reshape(3, 3)
        np.testing.assert_allclose([local.min(0), local.max(0)], record["bounds_local_m"], atol=2e-6)


def test_dynamics_record_contract_and_target_rejection(backend):
    backend.reset()
    target = np.array([.1, .1, -.1, .05, .04, -.03, .05]*2)
    backend.apply(Command(target, time.monotonic(), 0))
    for _ in range(120):
        backend.step()
    before = backend.read()
    assert before.sim_time == pytest.approx(4., abs=1e-10)
    np.testing.assert_allclose(before.position, target, atol=.025)
    assert before.extras["object"]["position_world"][2] == pytest.approx(.045, abs=.0005)
    assert backend.read().sim_time == before.sim_time
    previous = backend.data.ctrl.copy()
    target[6] = .1
    with pytest.raises(ValueError, match="bounds"):
        backend.apply(Command(target, time.monotonic(), 1))
    np.testing.assert_array_equal(backend.data.ctrl, previous)


def test_portable_bundle_needs_no_generated_cad_or_external_source(tmp_path, monkeypatch):
    import piper_sim.mujoco_asset as assets
    if not (ROOT/"assets/mujoco/manifest.json").exists():
        pytest.skip("Export the portable bundle first")
    shutil.copytree(ROOT/"assets/mujoco", tmp_path/"assets/mujoco")
    shutil.copytree(ROOT/"configs", tmp_path/"configs")
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    def forbid_rebuild(*args, **kwargs):
        pytest.fail("Portable runtime tried to rebuild original CAD")
    monkeypatch.setattr(assets, "build_model", forbid_rebuild)
    path, _ = assets.resolve_model(load())
    assert path == tmp_path/"assets/mujoco/workcell.xml"
    backend = MujocoBackend(load())
    try:
        assert backend.model.ncam == 3
        assert backend.model.nu == 16
        backend.step()
        assert backend.read().sim_time == pytest.approx(1/30)
    finally:
        backend.close()
    # Corrupted packaged geometry must fail closed instead of silently loading.
    mesh = next((tmp_path/"assets/mujoco/meshes").iterdir())
    mesh.write_bytes(mesh.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="Mesh changed"):
        assets.resolve_model(load())
