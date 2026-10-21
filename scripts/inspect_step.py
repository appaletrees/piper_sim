#!/usr/bin/env python3
"""Inspect AP203 assembly placements without a CAD runtime.

Handles the AXIS2_PLACEMENT_3D / ITEM_DEFINED_TRANSFORMATION representation used
by the supplied Creo file. This is an assembly metadata reader, not a mesher.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def inspect(path):
    entities = {int(k): v.replace("\n", "") for k, v in re.findall(
        r"#(\d+)\s*=\s*(.*?);", Path(path).read_text(), re.S)}
    refs = lambda value: [int(n) for n in re.findall(r"#(\d+)", value)]

    def coords(entity):
        return np.array([float(v) for v in re.search(
            r",\(([^()]*)\)\)", entities[entity]).group(1).split(",")])

    def frame(entity):
        p, z, x = refs(entities[entity])
        z, x = coords(z), coords(x)
        z /= np.linalg.norm(z)
        x -= x.dot(z)*z
        x /= np.linalg.norm(x)
        result = np.eye(4)
        result[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        result[:3, 3] = coords(p)
        return result

    placements = {}
    for value in entities.values():
        if value.startswith("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            relation, shape = refs(value)
            occurrence = refs(entities[shape])[-1]
            transform = int(re.search(
                r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(#(\d+)\)",
                entities[relation]).group(1))
            source, target = refs(entities[transform])
            placements[occurrence] = frame(target) @ np.linalg.inv(frame(source))
    products = {}
    children = {}
    for key, value in entities.items():
        if value.startswith("PRODUCT_DEFINITION("):
            formation = refs(value)[0]
            product = refs(entities[formation])[0]
            products[key] = re.search(r"PRODUCT\('([^']*)'", entities[product]).group(1)
        if value.startswith("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
            parent, child = refs(value)
            children.setdefault(parent, []).append((key, child))
    roots = set(children) - {child for values in children.values() for _, child in values}
    rows = []

    def walk(product, world, trail):
        name = products[product]
        rows.append({"product_id": product, "name": name, "path": trail + [name],
                     "transform_mm": np.round(world, 9).tolist(),
                     "has_children": product in children})
        for occurrence, child in children.get(product, []):
            walk(child, world @ placements[occurrence], trail + [name, str(occurrence)])
    for root in roots:
        walk(root, np.eye(4), [])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output", default="assets/generated/cad_assembly.json")
    args = parser.parse_args()
    rows = inspect(args.path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        if len(row["path"]) <= 3 or "REALSENSE" in row["name"]:
            print(row["name"], np.asarray(row["transform_mm"])[:3].tolist())
    print(f"{len(rows)} occurrences saved to {output}")
