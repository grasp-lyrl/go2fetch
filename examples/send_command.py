import argparse

from go2_interface.command import free_avoid, make_sport_client, move_for, stop


parser = argparse.ArgumentParser()
parser.add_argument("iface", help="network interface connected to Go2, e.g. enp2s0")

sub = parser.add_subparsers(dest="cmd", required=True)

sub.add_parser("stop")

avoid = sub.add_parser("avoid")
avoid.add_argument("state", choices=["on", "off"])

move = sub.add_parser("move")
move.add_argument("vx", type=float)
move.add_argument("vy", type=float)
move.add_argument("vyaw", type=float)
move.add_argument("--t", type=float, default=1.0)

args = parser.parse_args()
client = make_sport_client(args.iface)

try:
    if args.cmd == "stop":
        print(f"StopMove -> {stop(client)}")

    elif args.cmd == "avoid":
        print(f"FreeAvoid -> {free_avoid(client, args.state == 'on')}")

    elif args.cmd == "move":
        move_ret, stop_ret, vel, seconds = move_for(
            client, args.vx, args.vy, args.vyaw, args.t
        )
        print(f"Move -> ret={move_ret}, velocity={vel}, seconds={seconds}")
        print(f"StopMove -> ret={stop_ret}")

except KeyboardInterrupt:
    stop(client)
    