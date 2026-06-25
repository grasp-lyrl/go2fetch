import argparse
import time

from go2_interface.state import make_state_reader, state_to_dict


parser = argparse.ArgumentParser()
parser.add_argument("iface", help="network interface connected to Go2, e.g. enp2s0")
args = parser.parse_args()

read_state = make_state_reader(args.iface)

last_msg = None

while True:
    msg = read_state()

    if msg is None or msg is last_msg:
        time.sleep(0.001)
        continue

    last_msg = msg

    state = state_to_dict(msg)
    print(state["position"], state["velocity"], state["imu_state"]["rpy"])
    time.sleep(1)
    