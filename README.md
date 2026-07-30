# Go2 Fetch
https://grasp-lyrl.github.io/go2fetch/

## Installation

```bash
# setup python environment
git clone https://github.com/grasp-lyrl/go2fetch.git
cd go2fetch

# create python environment
python3.10 -m venv .venv
source .venv/bin/activate

# install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# install native video tools
sudo apt update
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
```

## Usage Example

Find the network interface connected to the Go2:

```bash
ip -br link
```

Read state:

```bash
# read state
python -m examples.read_state enp129s0

# stream camera snapshots to Rerun
python -m examples.read_camera enp129s0
```

Send commands:

```bash
# stop motion
python -m examples.send_command enp129s0 stop

# move forward for 5 seconds: vx vy vyaw
python -m examples.send_command enp129s0 move 0.10 0.0 0.0 --t 5.0

# rotate for 2 seconds
python -m examples.send_command enp129s0 move 0.0 0.0 0.25 --t 2.0
```

Replace `enp129s0` with your Go2 network interface.

## Rerun Recording & Visualization

We save recordings in `logs/` by default.

`logs/name.rrd` = state + lidar

`logs/name.mp4` = camera video

```bash
# record data
python ./scripts/record.py enp129s0 --out logs/go2fetch.rrd

# visualize a recorded bag
rerun logs/go2fetch.rrd

# visualize data live
python ./scripts/record.py enp129s0 --live

# visualize live and record concurrently
python ./scripts/record.py enp129s0 --live --out logs/go2fetch.rrd
```

Replace `enp129s0` with your Go2 network interface.
