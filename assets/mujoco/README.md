# Portable dual Piper MJCF

`workcell.xml` and `meshes/` load directly in MuJoCo 3.3.7, without Isaac, USD or the
external robot source directory. All mesh paths are relative. Lengths are metres.
`kinematics.urdf` is a mesh-free copy of the source articulation for FK validation;
it is not a second renderable robot asset.

The user's supplied STEP design defines dimensions and mounting transforms.
There are 99 CAD meshes (51 static parts, 20 wrist camera/mount parts and 28 gripper
parts), plus seven shared Piper STL link meshes. CAD geometry is triangulated at
the existing converter's 0.1 mm tolerance. The two wrists use D435i RGB optics;
the central camera uses D455 RGB optics. Camera optical metadata and every mesh
SHA256 are recorded in `manifest.json`.

There are 14 actuators: six arm joints and one `gripper` per side. Each gripper
actuator controls the full opening in metres through a fixed tendon; an equality
constraint couples its two physical finger joints. Do not independently command
`joint7` and `joint8`. The shared 14-value action/observation contract is unchanged.

Piper STL meshes and the source URDF are redistributed with `UPSTREAM_LICENSE`
copied unchanged from the supplied robot description. The CAD parts derive from
the user's supplied `demoa_asm_20261020.stp`; this bundle does not assign a new
license to that source design.

Rebuild after preparing the source CAD/URDF assets:

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/prepare_mujoco.py --output assets/mujoco
```

Use the project launcher for the shared control, recording and replay interfaces:

```bash
bash scripts/run_mujoco.sh --enable_cameras
```

Add `--interactive` to let the native Control sliders drive the project scene
and optionally record those actions with `--record`.

For a standalone model viewer:

```bash
.venv-mujoco/bin/python -m mujoco.viewer --mjcf=assets/mujoco/workcell.xml
```

Select the `home` keyframe in the standalone viewer for the configured 40 mm
gripper opening. The project launcher sets this initial state automatically.
See [MuJoCo runtime and validation](../../docs/mujoco.md) for dynamics assumptions.
