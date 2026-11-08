# Lab stage visual library

Authored in Blender 4.5 LTS by `scripts/blender/build_stage.py` from the original
workcell camera frames and reference-inspired laboratory appearance.

- `stage.blend`: editable source; Collections use world or named link coordinates.
- `meshes.npz`: exported vertices, triangle indices, normals and UVs.
- `manifest.json`: attachment parents, materials, provenance and source hashes.

Textures are shared with `../clean_table/textures` and packed in the blend file.
No collision shapes, masses, joints, camera intrinsics or control logic are
defined here. Camera covers/optical coatings, printed lettering, short cables
and the background room are visual approximations; original CAD stays intact.
The D435i stereo pair uses 50 mm spacing and the D455 pair uses 95 mm spacing.
Sensor origin/FOV continue to come from the existing camera rig, not these meshes.

Regenerate runtime exports through the build script after edits; arbitrary
manual changes in the blend file are not automatically converted to NPZ.
See `docs/visual_quality.md` for runtime captures, scope and acceptance criteria.
