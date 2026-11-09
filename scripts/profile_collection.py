#!/usr/bin/env python3
"""Run the unchanged demo CLI with opt-in wall-time/resource instrumentation."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--profile-output', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    if args.profile_output.exists():
        parser.error('Profile output already exists')
    from piper_sim.profiling import PipelineProfiler, instrument_collection, save_profile
    profiler = PipelineProfiler()
    code = None
    try:
        with profiler.stage('imports'):
            instrument_collection(profiler)
            spec = importlib.util.spec_from_file_location('profiled_demo', Path(__file__).with_name('demo_clean_table.py'))
            demo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo)
        sys.argv = [str(spec.origin), *forwarded]
        code = demo.main()
        return code
    finally:
        report = profiler.close()
        report.update(returncode=code, command=forwarded,
                      training_note='Instrumentation alone does not establish acceptance; see the closed-file audit.')
        save_profile(args.profile_output, report)


if __name__ == '__main__':
    raise SystemExit(main())
