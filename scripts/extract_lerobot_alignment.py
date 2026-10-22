#!/usr/bin/env python3
"""Export local reference images/states from the connector_plugin LeRobot dataset."""
import argparse
from pathlib import Path

from piper_sim.lerobot_alignment import extract_connector_plugin_alignment
from piper_sim.contracts import GRIPPER_MAX_WIDTH_M


def parse_sample(value):
    try:
        episode, frame = (int(v) for v in value.split(":"))
        if min(episode, frame) < 0:
            raise ValueError
        return episode, frame
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use zero-based EPISODE:FRAME, e.g. 0:200") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--sample", type=parse_sample, action="append", required=True)
    parser.add_argument("--gripper-limit-m", type=float, default=GRIPPER_MAX_WIDTH_M,
                        help="Report violations of this sim limit; never clips measured values")
    parser.add_argument("--source-root", type=Path,
                        help="Optional example_agilex checkout to hash producer evidence")
    args = parser.parse_args()
    manifest = extract_connector_plugin_alignment(
        args.dataset, args.output, args.sample, gripper_limit_m=args.gripper_limit_m,
        source_root=args.source_root)
    violations = sum(len(row["state_bounds_violations"]) for row in manifest["samples"])
    print(f"Exported {len(manifest['samples'])} samples, {violations} state bound violations retained")
    print(args.output.resolve() / "manifest.json")


if __name__ == "__main__":
    main()
