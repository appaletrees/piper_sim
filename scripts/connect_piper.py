#!/usr/bin/env python3
"""Connect to real dual Piper arms, with no simulator; default is read-only."""
import argparse
import json
import math
import time

from piper_sim.config import load
from piper_sim.piper_connection import PiperConnection, inspect_can_interfaces, state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--can-left")
    parser.add_argument("--can-right")
    parser.add_argument("--list-can", action="store_true", help="Only list existing SocketCAN interfaces")
    parser.add_argument("--samples", type=int, default=1, help="Read-only samples; 0 continues until Ctrl+C")
    parser.add_argument("--rate", type=float, default=30.)
    parser.add_argument("--timeout", type=float, default=5.)
    parser.add_argument("--command-port", type=int, help="Loopback HTTP /state and /target")
    parser.add_argument("--allow-actuation", action="store_true")
    args = parser.parse_args()
    if args.list_can:
        print(json.dumps(inspect_can_interfaces(), indent=2))
        return
    if not math.isfinite(args.rate) or args.rate <= 0 or args.samples < 0:
        parser.error("rate must be positive and samples must be nonnegative")
    if args.command_port is not None and not 1 <= args.command_port <= 65535:
        parser.error("command-port must be between 1 and 65535")
    if args.allow_actuation and not args.command_port:
        parser.error("--allow-actuation requires a continuous --command-port stream")
    cfg = load(args.config)
    server = None
    try:
        with PiperConnection(cfg["hardware"], can_left=args.can_left, can_right=args.can_right,
                             timeout_s=args.timeout, allow_actuation=args.allow_actuation) as robot:
            if args.command_port:
                from piper_sim.control import CommandServer
                server = CommandServer(args.command_port, allow_commands=args.allow_actuation)
                print(f"Connected: http://127.0.0.1:{args.command_port}/state", flush=True)
            index = 0
            last_sequence = -1
            while server or args.samples == 0 or index < args.samples:
                start = time.monotonic()
                observation = robot.read()
                if server:
                    server.publish(observation)
                    incoming = server.sample(cfg["hardware"]["command_max_age"]) if args.allow_actuation else None
                    if incoming is not None and incoming.sequence != last_sequence:
                        if not robot.armed:
                            robot.arm()
                        # Keep the sender's original timestamp/sequence: resending a
                        # stale command as "new" would hide a stopped control source.
                        robot.apply(incoming)
                        last_sequence = incoming.sequence
                else:
                    print(json.dumps(state_dict(observation)), flush=True)
                index += 1
                time.sleep(max(0., 1/args.rate-(time.monotonic()-start)))
    except KeyboardInterrupt:
        pass
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Piper connection stopped: {error}\n")
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
