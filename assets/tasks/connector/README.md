# Connector board asset

Generated from the user's original `scene/scene1_ peg in hole/接插件原型板1.STL` by
`scripts/prepare_connector.py`. The SLDPRT and reference photos remain in that source directory.

The source STL uses 170 × 33 × 60 coordinates; interpreted as mm and transformed to
170 × 60 × 33 mm with +Z up. `manifest.json` records the transform, six hole polygons,
source hash, exact visual mesh and 22 convex collision pieces. No hole geometry is fitted to a photo.

Convex decomposition follows the source's extrusion layers; the combined volume is
checked against the signed tetrahedral volume of the original STL. Each engine uses
the same collision pieces, so holes are preserved rather than filled by a single convex hull.

Plug geometry is independent and generated from the estimated dimensions in
`configs/tasks/connector.yaml`. Runtime needs no CAD/shapely installation. See
`docs/connector_scene.md` for the provenance, limits and validation commands.
