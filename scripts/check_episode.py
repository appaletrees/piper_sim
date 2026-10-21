#!/usr/bin/env python3
import argparse
import json
from piper_sim.dataset import validate_episode

parser = argparse.ArgumentParser()
parser.add_argument("episode")
args = parser.parse_args()
print(json.dumps(validate_episode(args.episode), indent=2, ensure_ascii=False))
