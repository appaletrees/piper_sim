#!/usr/bin/env python3
"""Optionally convert completed simulation HDF5 episodes into a local LeRobot v2.1 dataset."""
import argparse
import json
from piper_sim.lerobot_export import export_lerobot_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',required=True,help='Completed .h5 files or a directory (sorted *.h5)')
    parser.add_argument('--output',required=True,help='New local dataset directory; existing paths are never overwritten')
    parser.add_argument('--image-size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'),
                        help='Optional aspect-preserving resize with black padding; default keeps original resolution')
    args=parser.parse_args()
    result=export_lerobot_v2(args.input,args.output,image_size=args.image_size,progress=lambda s:print(s,flush=True))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
