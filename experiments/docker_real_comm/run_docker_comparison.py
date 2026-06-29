from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.mpc.yml"
GPU_COMPOSE = ROOT / "docker" / "docker-compose.gpu.yml"
SEQ = 4
HIDDEN = 16
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN = 32


def run(cmd: list[str], env: dict[str, str], timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def docker_bind_root(out_host: Path, timestamp: str) -> Path:
    override = os.environ.get("PHANTOM_DOCKER_BIND_HOST")
    if override:
        path = Path(override)
    elif os.name == "nt":
        # Docker Desktop can intermittently fail to bind-mount non-system drives
        # (for example F:) as /run/desktop/mnt/host/<drive>. Stage shared files
        # on the default temp drive and copy them back into the experiment folder.
        path = Path(tempfile.gettempdir()) / "phantomshield_docker_out" / timestamp
    else:
        path = out_host
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_done(text: str) -> float:
    match = re.search(r"done:\s+([0-9.]+)s", text)
    if not match:
        raise RuntimeError(f"could not parse elapsed from:\n{text}")
    return float(match.group(1))


def parse_timing(text: str) -> dict[str, float]:
    timing = {}
    for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", text):
        timing[key] = float(value)
    for key, value in re.findall(r"(send_msgs|recv_msgs|send_bytes|recv_bytes)=([0-9]+)", text):
        timing[key] = int(value)
    return timing


def parse_role_timing(text: str, role: str) -> dict:
    match = re.search(rf"\[{re.escape(role)}\] timing ([^\n]+)", text)
    if not match:
        return {}
    return parse_timing(match.group(1))


def parse_role_breakdown(text: str, role_id: int) -> dict:
    match = re.search(rf"\[P{role_id}\] timing_breakdown ([^\n]+)", text)
    if not match:
        return {}
    return parse_timing(match.group(1))


def compose_base(project: str, use_gpu: bool = False) -> list[str]:
    cmd = ["docker", "compose", "-p", project, "-f", str(COMPOSE)]
    if use_gpu:
        cmd.extend(["-f", str(GPU_COMPOSE)])
    return cmd


def cleanup(project: str, env: dict[str, str], use_gpu: bool = False) -> None:
    run([*compose_base(project, use_gpu), "down", "--remove-orphans"], env, check=False, timeout=120)


def mcu_env(base_env: dict[str, str], out_host: Path, case_dir: str, spec: dict) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "PHANTOM_OUT_HOST": str(out_host),
            "MCU_KIND": spec["mcu_kind"],
            "MCU_OP": spec["mcu_op"],
            "MCU_PORT": str(spec.get("mcu_port", 9200 if spec["mcu_kind"] == "tensor" else 9300)),
            "MCU_LEN": str(spec.get("len", 64)),
            "MCU_M": str(spec.get("m", 4)),
            "MCU_K": str(spec.get("k", 16)),
            "MCU_N": str(spec.get("n", 16)),
            "MCU_OUT_DIR": f"/workspace/out/{case_dir}",
            "MCU_START_DELAY": "1",
            "MCU_CUDA_MIN_OPS": str(spec.get("cuda_min_ops", base_env.get("MCU_CUDA_MIN_OPS", 1048576))),
        }
    )
    return env


def crypten_env(base_env: dict[str, str], out_host: Path, case_dir: str, spec: dict, repeat: int) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "PHANTOM_OUT_HOST": str(out_host),
            "CRYPTEN_OP": spec["crypten_op"],
            "CRYPTEN_REPEAT": str(repeat),
            "CRYPTEN_LEN": str(spec.get("len", 64)),
            "CRYPTEN_M": str(spec.get("m", 4)),
            "CRYPTEN_K": str(spec.get("k", 16)),
            "CRYPTEN_N": str(spec.get("n", 16)),
            "CRYPTEN_BATCH": str(spec.get("batch", 1)),
            "CRYPTEN_SOFTMAX_ROWS": str(spec.get("softmax_rows", 16)),
            "CRYPTEN_SOFTMAX_COLS": str(spec.get("softmax_cols", 4)),
            "CRYPTEN_OUT_DIR": f"/workspace/out/{case_dir}",
            "CRYPTEN_START_DELAY": "1",
        }
    )
    return env


def run_mcu_once(project: str, env: dict[str, str], log_dir: Path, use_gpu: bool) -> tuple[dict, str]:
    cleanup(project, env, use_gpu)
    log_dir.mkdir(parents=True, exist_ok=True)
    p0_out = p0_err = p1_out = p1_err = hp_logs = ""
    p0 = subprocess.CompletedProcess([], 1)
    p1 = subprocess.CompletedProcess([], 1)
    verify = subprocess.CompletedProcess([], 1, "", "")
    try:
        run([*compose_base(project, use_gpu), "up", "-d", "mcu-hp"], env, timeout=300)
        time.sleep(1.0)
        p0 = subprocess.Popen(
            [*compose_base(project, use_gpu), "run", "--rm", "mcu-p0"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p1 = subprocess.Popen(
            [*compose_base(project, use_gpu), "run", "--rm", "mcu-p1"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p0_out, p0_err = p0.communicate(timeout=300)
        p1_out, p1_err = p1.communicate(timeout=300)
        hp_logs = run([*compose_base(project, use_gpu), "logs", "--no-color", "mcu-hp"], env, check=False, timeout=120).stdout
        verify = run([*compose_base(project, use_gpu), "run", "--rm", "mcu-verify"], env, check=False, timeout=300)
    finally:
        cleanup(project, env, use_gpu)

    (log_dir / "p0.log").write_text(p0_out + p0_err, encoding="utf-8")
    (log_dir / "p1.log").write_text(p1_out + p1_err, encoding="utf-8")
    (log_dir / "hp.log").write_text(hp_logs, encoding="utf-8")
    (log_dir / "verify.log").write_text(verify.stdout + verify.stderr, encoding="utf-8")

    if p0.returncode != 0 or p1.returncode != 0:
        raise RuntimeError(f"MCU role failed p0={p0.returncode} p1={p1.returncode}\n{p0_out}{p0_err}\n{p1_out}{p1_err}")
    if verify.returncode != 0:
        raise RuntimeError(f"MCU verify failed:\n{verify.stdout}{verify.stderr}")
    role_times = {
        "p0": parse_done(p0_out),
        "p1": parse_done(p1_out),
        "hp": parse_done(hp_logs),
    }
    role_timings = {
        "p0": parse_role_timing(p0_out, "p0"),
        "p1": parse_role_timing(p1_out, "p1"),
        "hp": parse_role_timing(hp_logs, "hp"),
    }
    role_timings["p0"].update(parse_role_breakdown(p0_out, 0))
    role_timings["p1"].update(parse_role_breakdown(p1_out, 1))
    max_role = max(role_times, key=role_times.get)
    aggregate = {
        "elapsed_s": role_times[max_role],
        "critical_role": max_role,
        "p0_elapsed_s": role_times["p0"],
        "p1_elapsed_s": role_times["p1"],
        "hp_elapsed_s": role_times["hp"],
    }
    for role, timing in role_timings.items():
        for key, value in timing.items():
            aggregate[f"{role}_{key}"] = value
    return aggregate, verify.stdout.strip()


def run_crypten_once(project: str, env: dict[str, str], out_host: Path, case_dir: str, log_dir: Path) -> dict:
    cleanup(project, env)
    log_dir.mkdir(parents=True, exist_ok=True)
    r1 = subprocess.CompletedProcess([], 1, "", "")
    r0_logs = ""
    try:
        run([*compose_base(project), "up", "-d", "crypten-r0"], env, timeout=300)
        time.sleep(1.0)
        r1 = run([*compose_base(project), "run", "--rm", "crypten-r1"], env, check=False, timeout=600)
        r0_logs = run([*compose_base(project), "logs", "--no-color", "crypten-r0"], env, check=False, timeout=120).stdout
    finally:
        cleanup(project, env)

    (log_dir / "rank0.log").write_text(r0_logs, encoding="utf-8")
    (log_dir / "rank1.log").write_text(r1.stdout + r1.stderr, encoding="utf-8")

    if r1.returncode != 0:
        raise RuntimeError(f"CrypTen rank1 failed:\n{r1.stdout}{r1.stderr}")
    rank0 = json.loads((out_host / case_dir / "rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_host / case_dir / "rank1.json").read_text(encoding="utf-8"))
    return {
        "elapsed_s": max(float(rank0["median_s"]), float(rank1["median_s"])),
        "rank0_elapsed_s": float(rank0["median_s"]),
        "rank1_elapsed_s": float(rank1["median_s"]),
        "plain_median_s": float(rank0["plain_median_s"]) if rank0.get("plain_median_s") is not None else 0.0,
    }


def shape_config(preset: str) -> tuple[int, int, int, int]:
    if preset == "toy":
        return 4, 16, 4, 32
    if preset == "bert":
        return 128, 768, 12, 3072
    if preset == "large":
        return 128, 1024, 16, 4096
    raise ValueError(f"unknown preset: {preset}")


def specs_for_batches(batches: list[int], preset: str, modules: set[str]) -> list[dict]:
    seq, hidden, heads, _ffn = shape_config(preset)
    specs = []
    for batch in batches:
        elem_len = batch * seq * hidden
        if "elemul" in modules:
            specs.append(
                {
                    "module": "elemul",
                    "batch": batch,
                    "shape": f"[{elem_len}]",
                    "mcu_kind": "tensor",
                    "mcu_op": "elemul",
                    "crypten_op": "elemul",
                    "len": elem_len,
                }
            )
        if "matmul" in modules:
            m, k, n = batch * seq, hidden, hidden
            specs.append(
                {
                    "module": "matmul",
                    "batch": batch,
                    "shape": f"[{m},{k}]x[{k},{n}]",
                    "mcu_kind": "tensor",
                    "mcu_op": "matmul",
                    "crypten_op": "matmul",
                    "m": m,
                    "k": k,
                    "n": n,
                }
            )
        nl_n = batch * seq * hidden
        for op in ["exp", "sigmoid", "gelu"]:
            if op in modules:
                specs.append(
                    {
                        "module": op,
                        "batch": batch,
                        "shape": f"[{nl_n}]",
                        "mcu_kind": "nonlinear",
                        "mcu_op": op,
                        "crypten_op": op,
                        "n": nl_n,
                        "len": nl_n,
                        "mcu_port": 9300,
                    }
                )
        if "softmax" in modules:
            rows = batch * heads * seq
            specs.append(
                {
                    "module": "softmax",
                    "batch": batch,
                    "shape": f"[{rows},{seq}]",
                    "mcu_kind": "nonlinear",
                    "mcu_op": "softmax",
                    "crypten_op": "softmax",
                    "n": rows,
                    "k": seq,
                    "softmax_rows": rows,
                    "softmax_cols": seq,
                    "mcu_port": 9300,
                }
            )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--preset", choices=["toy", "bert", "large"], default="toy")
    parser.add_argument("--modules", default="elemul,matmul,exp,sigmoid,gelu,softmax")
    parser.add_argument("--mcu-gpu", action="store_true")
    parser.add_argument("--mcu-cuda-min-ops", type=int, default=1048576)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_docker_real_comm"
    out_host = out / "shared"
    bind_host = docker_bind_root(out_host, timestamp)
    logs = out / "logs"
    out_host.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PHANTOM_OUT_HOST"] = str(bind_host)
    base_env["MCU_CUDA_MIN_OPS"] = str(args.mcu_cuda_min_ops)
    if args.mcu_gpu:
        base_env["MCU_DOCKERFILE"] = "docker/Dockerfile.mcu.cuda"
        base_env["MCU_IMAGE"] = "phantomshield-mcu:cuda"

    if not args.skip_build:
        print("[docker-compare] building images")
        run([*compose_base("phantomshield_build", args.mcu_gpu), "build", "mcu-hp", "crypten-r0"], base_env, timeout=1800)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    modules = {x.strip() for x in args.modules.split(",") if x.strip()}
    rows = []
    for spec_index, spec in enumerate(specs_for_batches(batches, args.preset, modules)):
        print(f"[docker-compare] {spec['module']} b={spec['batch']} shape={spec['shape']}")
        mcu_times = []
        crypten_times = []
        plain_times = []
        mcu_records = []
        crypten_records = []
        verify = ""
        for rep in range(args.repeat):
            slug = f"{spec_index:02d}_{spec['module']}_b{spec['batch']}_r{rep}"
            project_mcu = f"psmcu{spec_index:02d}{rep}"
            project_crypten = f"psct{spec_index:02d}{rep}"
            m_env = mcu_env(base_env, bind_host, f"{slug}_mcu", spec)
            c_env = crypten_env(base_env, bind_host, f"{slug}_crypten", spec, 1)
            m_record, verify = run_mcu_once(project_mcu, m_env, logs / slug / "mcu", args.mcu_gpu)
            c_record = run_crypten_once(project_crypten, c_env, bind_host, f"{slug}_crypten", logs / slug / "crypten")
            mcu_records.append(m_record)
            crypten_records.append(c_record)
            mcu_times.append(float(m_record["elapsed_s"]))
            crypten_times.append(float(c_record["elapsed_s"]))
            plain_times.append(float(c_record["plain_median_s"]))
        mcu_median = statistics.median(mcu_times)
        crypten_median = statistics.median(crypten_times)
        plain_median = statistics.median(plain_times)
        median_mcu_record = sorted(mcu_records, key=lambda r: float(r["elapsed_s"]))[len(mcu_records) // 2]
        median_crypten_record = sorted(crypten_records, key=lambda r: float(r["elapsed_s"]))[len(crypten_records) // 2]
        rows.append(
            {
                "module": spec["module"],
                "batch_size": spec["batch"],
                "shape": spec["shape"],
                "measurement_kind": "docker_real_comm_mcu_3role_tcp_vs_crypten_2rank_gloo",
                "preset": args.preset,
                "mcu_gpu": args.mcu_gpu,
                "mcu_cuda_min_ops": args.mcu_cuda_min_ops,
                "mcu_docker_median_s": f"{mcu_median:.9f}",
                "crypten_docker_median_s": f"{crypten_median:.9f}",
                "torch_plain_median_s": f"{plain_median:.9f}",
                "ratio_mcu_over_crypten": f"{mcu_median / crypten_median:.6f}",
                "ratio_mcu_over_plain": f"{mcu_median / plain_median:.6f}" if plain_median else "",
                "ratio_crypten_over_plain": f"{crypten_median / plain_median:.6f}" if plain_median else "",
                "mcu_critical_role": median_mcu_record.get("critical_role", ""),
                "mcu_p0_elapsed_s": f"{float(median_mcu_record.get('p0_elapsed_s', 0.0)):.9f}",
                "mcu_p1_elapsed_s": f"{float(median_mcu_record.get('p1_elapsed_s', 0.0)):.9f}",
                "mcu_hp_elapsed_s": f"{float(median_mcu_record.get('hp_elapsed_s', 0.0)):.9f}",
                "mcu_critical_comm_s": f"{float(median_mcu_record.get(median_mcu_record.get('critical_role', '') + '_comm_s', 0.0)):.9f}",
                "mcu_critical_local_s": f"{float(median_mcu_record.get(median_mcu_record.get('critical_role', '') + '_local_s', 0.0)):.9f}",
                "mcu_critical_protocol_s": f"{float(median_mcu_record.get(median_mcu_record.get('critical_role', '') + '_protocol_s', 0.0)):.9f}",
                "mcu_critical_write_s": f"{float(median_mcu_record.get(median_mcu_record.get('critical_role', '') + '_write_s', 0.0)):.9f}",
                "mcu_p0_comm_s": f"{float(median_mcu_record.get('p0_comm_s', 0.0)):.9f}",
                "mcu_p1_comm_s": f"{float(median_mcu_record.get('p1_comm_s', 0.0)):.9f}",
                "mcu_hp_comm_s": f"{float(median_mcu_record.get('hp_comm_s', 0.0)):.9f}",
                "mcu_p0_recv_wait_s": f"{float(median_mcu_record.get('p0_recv_wait_s', 0.0)):.9f}",
                "mcu_p1_recv_wait_s": f"{float(median_mcu_record.get('p1_recv_wait_s', 0.0)):.9f}",
                "mcu_hp_recv_wait_s": f"{float(median_mcu_record.get('hp_recv_wait_s', 0.0)):.9f}",
                "mcu_p0_recv_read_s": f"{float(median_mcu_record.get('p0_recv_read_s', 0.0)):.9f}",
                "mcu_p1_recv_read_s": f"{float(median_mcu_record.get('p1_recv_read_s', 0.0)):.9f}",
                "mcu_hp_recv_read_s": f"{float(median_mcu_record.get('hp_recv_read_s', 0.0)):.9f}",
                "mcu_p0_local_s": f"{float(median_mcu_record.get('p0_local_s', 0.0)):.9f}",
                "mcu_p1_local_s": f"{float(median_mcu_record.get('p1_local_s', 0.0)):.9f}",
                "mcu_hp_local_s": f"{float(median_mcu_record.get('hp_local_s', 0.0)):.9f}",
                "mcu_p0_compute_s": f"{float(median_mcu_record.get('p0_compute_s', 0.0)):.9f}",
                "mcu_p1_compute_s": f"{float(median_mcu_record.get('p1_compute_s', 0.0)):.9f}",
                "mcu_hp_compute_s": f"{float(median_mcu_record.get('hp_compute_s', 0.0)):.9f}",
                "mcu_p0_other_local_s": f"{float(median_mcu_record.get('p0_other_local_s', 0.0)):.9f}",
                "mcu_p1_other_local_s": f"{float(median_mcu_record.get('p1_other_local_s', 0.0)):.9f}",
                "mcu_hp_other_local_s": f"{float(median_mcu_record.get('hp_other_local_s', 0.0)):.9f}",
                "mcu_p0_protocol_s": f"{float(median_mcu_record.get('p0_protocol_s', 0.0)):.9f}",
                "mcu_p1_protocol_s": f"{float(median_mcu_record.get('p1_protocol_s', 0.0)):.9f}",
                "mcu_p0_write_s": f"{float(median_mcu_record.get('p0_write_s', 0.0)):.9f}",
                "mcu_p1_write_s": f"{float(median_mcu_record.get('p1_write_s', 0.0)):.9f}",
                "mcu_p0_send_msgs": int(median_mcu_record.get("p0_send_msgs", 0)),
                "mcu_p0_recv_msgs": int(median_mcu_record.get("p0_recv_msgs", 0)),
                "mcu_p1_send_msgs": int(median_mcu_record.get("p1_send_msgs", 0)),
                "mcu_p1_recv_msgs": int(median_mcu_record.get("p1_recv_msgs", 0)),
                "mcu_hp_send_msgs": int(median_mcu_record.get("hp_send_msgs", 0)),
                "mcu_hp_recv_msgs": int(median_mcu_record.get("hp_recv_msgs", 0)),
                "mcu_p0_send_bytes": int(median_mcu_record.get("p0_send_bytes", 0)),
                "mcu_p0_recv_bytes": int(median_mcu_record.get("p0_recv_bytes", 0)),
                "mcu_p1_send_bytes": int(median_mcu_record.get("p1_send_bytes", 0)),
                "mcu_p1_recv_bytes": int(median_mcu_record.get("p1_recv_bytes", 0)),
                "mcu_hp_send_bytes": int(median_mcu_record.get("hp_send_bytes", 0)),
                "mcu_hp_recv_bytes": int(median_mcu_record.get("hp_recv_bytes", 0)),
                "mcu_p0_cpu_matmul_calls": int(median_mcu_record.get("p0_cpu_matmul_calls", 0)),
                "mcu_p1_cpu_matmul_calls": int(median_mcu_record.get("p1_cpu_matmul_calls", 0)),
                "mcu_hp_cpu_matmul_calls": int(median_mcu_record.get("hp_cpu_matmul_calls", 0)),
                "mcu_p0_cuda_matmul_calls": int(median_mcu_record.get("p0_cuda_matmul_calls", 0)),
                "mcu_p1_cuda_matmul_calls": int(median_mcu_record.get("p1_cuda_matmul_calls", 0)),
                "mcu_hp_cuda_matmul_calls": int(median_mcu_record.get("hp_cuda_matmul_calls", 0)),
                "mcu_p0_fused_party_calls": int(median_mcu_record.get("p0_fused_party_calls", 0)),
                "mcu_p1_fused_party_calls": int(median_mcu_record.get("p1_fused_party_calls", 0)),
                "mcu_hp_fused_hp_calls": int(median_mcu_record.get("hp_fused_hp_calls", 0)),
                "mcu_p0_cuda_fallbacks": int(median_mcu_record.get("p0_cuda_fallbacks", 0)),
                "mcu_p1_cuda_fallbacks": int(median_mcu_record.get("p1_cuda_fallbacks", 0)),
                "mcu_hp_cuda_fallbacks": int(median_mcu_record.get("hp_cuda_fallbacks", 0)),
                "crypten_rank0_median_s": f"{float(median_crypten_record.get('rank0_elapsed_s', 0.0)):.9f}",
                "crypten_rank1_median_s": f"{float(median_crypten_record.get('rank1_elapsed_s', 0.0)):.9f}",
                "mcu_times_s": ";".join(f"{t:.9f}" for t in mcu_times),
                "crypten_times_s": ";".join(f"{t:.9f}" for t in crypten_times),
                "plain_times_s": ";".join(f"{t:.9f}" for t in plain_times),
                "verify": verify.replace("\n", " | "),
            }
        )

    write_csv(out / "docker_real_comm_comparison.csv", rows)
    write_csv(
        out / "summary.csv",
        [
            {
                "module": r["module"],
                "preset": r["preset"],
                "batch_size": r["batch_size"],
                "shape": r["shape"],
                "ratio_mcu_over_crypten": r["ratio_mcu_over_crypten"],
                "ratio_mcu_over_plain": r["ratio_mcu_over_plain"],
                "ratio_crypten_over_plain": r["ratio_crypten_over_plain"],
                "mcu_docker_median_s": r["mcu_docker_median_s"],
                "crypten_docker_median_s": r["crypten_docker_median_s"],
                "torch_plain_median_s": r["torch_plain_median_s"],
                "mcu_critical_role": r["mcu_critical_role"],
                "mcu_critical_comm_s": r["mcu_critical_comm_s"],
                "mcu_critical_local_s": r["mcu_critical_local_s"],
                "mcu_critical_protocol_s": r["mcu_critical_protocol_s"],
                "mcu_critical_write_s": r["mcu_critical_write_s"],
                "mcu_p0_compute_s": r["mcu_p0_compute_s"],
                "mcu_p1_compute_s": r["mcu_p1_compute_s"],
                "mcu_hp_compute_s": r["mcu_hp_compute_s"],
                "mcu_p0_recv_wait_s": r["mcu_p0_recv_wait_s"],
                "mcu_p1_recv_wait_s": r["mcu_p1_recv_wait_s"],
                "mcu_hp_recv_wait_s": r["mcu_hp_recv_wait_s"],
                "mcu_p0_recv_read_s": r["mcu_p0_recv_read_s"],
                "mcu_p1_recv_read_s": r["mcu_p1_recv_read_s"],
                "mcu_hp_recv_read_s": r["mcu_hp_recv_read_s"],
            }
            for r in rows
        ],
    )
    ratios = [float(r["ratio_mcu_over_crypten"]) for r in rows]
    summary = [
        "# Docker real-communication summary",
        "",
        f"- Output: `{out}`",
        f"- Rows: {len(rows)}",
        f"- Preset: `{args.preset}`",
        f"- MCU GPU: `{args.mcu_gpu}`",
        f"- Median ratio mcu/crypten: {statistics.median(ratios):.3f}x",
        f"- Min ratio: {min(ratios):.3f}x",
        f"- Max ratio: {max(ratios):.3f}x",
        "- MCU timing columns split socket send/recv time from local role time.",
        "- Torch plaintext timing is measured in the CrypTen container on rank 0.",
        "",
        "MCU uses three role containers over TCP. CrypTen uses two rank containers over Gloo/TCP.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    if bind_host.resolve() != out_host.resolve():
        if out_host.exists():
            shutil.rmtree(out_host)
        shutil.copytree(bind_host, out_host)
    print(f"[docker-compare] wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
