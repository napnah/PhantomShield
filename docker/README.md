# Docker MPC experiments

This directory provides containerized role runners for two real-communication
benchmark paths:

- MCU Rust: three containers, `mcu-hp`, `mcu-p0`, and `mcu-p1`, connected over a
  Docker bridge network with TCP.
- CrypTen: two containers, `crypten-r0` and `crypten-r1`, connected with
  `crypten.init()` through PyTorch Gloo over a Docker bridge network.

Run the automated comparison from the repository root:

```powershell
python experiments\docker_real_comm\run_docker_comparison.py --repeat 3 --batches 1,2,4
```

The runner builds the images, starts the role containers, verifies MCU outputs,
and writes CSV files under a timestamped `experiments/<timestamp>_docker_real_comm`
directory.

Manual smoke examples:

```powershell
$env:PHANTOM_OUT_HOST = "$PWD\experiments\docker_runs"
$env:MCU_KIND = "tensor"
$env:MCU_OP = "elemul"
$env:MCU_LEN = "64"
$env:MCU_OUT_DIR = "/workspace/out/manual_mcu"
docker compose -f docker\docker-compose.mpc.yml up -d mcu-hp
docker compose -f docker\docker-compose.mpc.yml run --rm mcu-p0
docker compose -f docker\docker-compose.mpc.yml run --rm mcu-p1
docker compose -f docker\docker-compose.mpc.yml run --rm mcu-verify
docker compose -f docker\docker-compose.mpc.yml down
```
