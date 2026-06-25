# Go2 Fetch

### Installation

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
```

```markdown
### Usage Example

Find the network interface connected to the Go2:

```bash
ip -br link
```

Read state:

```bash
python -m examples.read_state enp129s0
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

