from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.mpc.yml"
WARM_COMPOSE = ROOT / "docker" / "docker-compose.warm.yml"
EXPORT_SHARES = ROOT / "experiments" / "docker_bert_full" / "export_bert_session_shares.py"
INPUT_SHARE_SERVICE = ROOT / "experiments" / "docker_bert_full" / "mcu_input_share_service.py"
PROJECT = os.environ.get("AEGISX_WARM_PROJECT", "aegisxwarm")
DEFAULT_WARM_ROOT = ROOT / ".local" / "docker_warm"
DEFAULT_CACHE_TEXT = "This movie is wonderful and heartwarming."
HEARTBEAT_STALE_S = float(os.environ.get("AEGISX_SERVICE_HEARTBEAT_STALE_S", "30"))
SERVICE_REQUEST_TIMEOUT_S = float(os.environ.get("AEGISX_SERVICE_REQUEST_TIMEOUT_S", "45"))


def run(cmd: list[str], env: dict[str, str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def compose() -> list[str]:
    return ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE), "-f", str(WARM_COMPOSE)]


def warm_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("AEGISX_WARM_ROOT", DEFAULT_WARM_ROOT))
    out = Path(os.environ.get("AEGISX_WARM_OUT_HOST", root / "out"))
    shares = Path(os.environ.get("AEGISX_WARM_SHARES_HOST", root / "bert_shares"))
    out.mkdir(parents=True, exist_ok=True)
    shares.mkdir(parents=True, exist_ok=True)
    return out, shares


def service_dir() -> Path:
    out, _ = warm_paths()
    path = out / "crypten_service"
    path.mkdir(parents=True, exist_ok=True)
    (path / "requests").mkdir(parents=True, exist_ok=True)
    (path / "responses").mkdir(parents=True, exist_ok=True)
    return path


def mcu_service_dir() -> Path:
    out, _ = warm_paths()
    path = out / "mcu_service"
    path.mkdir(parents=True, exist_ok=True)
    (path / "requests").mkdir(parents=True, exist_ok=True)
    (path / "responses").mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def file_age_s(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def heartbeat_fresh(path: Path, stale_s: float = HEARTBEAT_STALE_S) -> bool:
    age = file_age_s(path)
    return age is not None and age <= stale_s


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def service_config_matches(ready: dict, max_seq_len: int, layers: int) -> bool:
    return int(ready.get("max_seq_len", -1)) == int(max_seq_len) and int(ready.get("max_layers", -1)) == int(layers)


def mcu_model_cache_id(max_seq_len: int, layers: int, scale_bits: int = 16) -> str:
    return f"model_l{layers}_s{max_seq_len}_q{scale_bits}"


def mcu_model_cache_dir(max_seq_len: int, layers: int, scale_bits: int = 16) -> Path:
    _, shares = warm_paths()
    return shares / "_model_cache" / mcu_model_cache_id(max_seq_len, layers, scale_bits)


def warm_env() -> dict[str, str]:
    out, shares = warm_paths()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MCU_IMAGE", env.get("MCU_DASHBOARD_IMAGE", "phantomshield-mcu:bert-session-fixed"))
    env.setdefault("CRYPTEN_IMAGE", "phantomshield-crypten:local")
    env["AEGISX_WARM_OUT_HOST"] = str(out)
    env["AEGISX_WARM_SHARES_HOST"] = str(shares)
    return env


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def start() -> dict:
    env = warm_env()
    t0 = time.perf_counter()
    result = run([*compose(), "up", "-d", "mcu-warm-hp", "mcu-warm-p0", "mcu-warm-p1", "crypten-warm-r0", "crypten-warm-r1"], env, timeout=300, check=False)
    elapsed = time.perf_counter() - t0
    return {
        "success": result.returncode == 0,
        "project": PROJECT,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def stop() -> dict:
    stop_crypten_service()
    env = warm_env()
    result = run([*compose(), "down", "--remove-orphans"], env, timeout=180, check=False)
    return {"success": result.returncode == 0, "project": PROJECT, "stdout": result.stdout, "stderr": result.stderr}


def status() -> dict:
    env = warm_env()
    result = run([*compose(), "ps", "--format", "json"], env, timeout=60, check=False)
    services = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            services.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {
        "success": result.returncode == 0,
        "project": PROJECT,
        "services": services,
        "running": [s.get("Service") for s in services if str(s.get("State", "")).lower() == "running"],
        "stdout": result.stdout if not services else "",
        "stderr": result.stderr,
    }


def ensure_running() -> None:
    state = status()
    required = {"mcu-warm-hp", "mcu-warm-p0", "mcu-warm-p1", "crypten-warm-r0", "crypten-warm-r1"}
    running = set(state.get("running", []))
    if not required.issubset(running):
        started = start()
        if not started.get("success"):
            raise RuntimeError(f"warm service start failed:\n{started.get('stdout')}\n{started.get('stderr')}")


def _crypten_service_common_env(max_seq_len: int = 16, layers: int = 12) -> list[str]:
    return [
        "CRYPTEN_OP=bert_full",
        f"CRYPTEN_MAX_SEQ_LEN={max_seq_len}",
        f"CRYPTEN_MAX_LAYERS={layers}",
        "CRYPTEN_BERT_NONLINEAR=native",
        "CRYPTEN_SERVICE_DIR=/workspace/out/crypten_service",
        "RENDEZVOUS=tcp://crypten-warm-r0:29500",
        "DISTRIBUTED_BACKEND=gloo",
        "WORLD_SIZE=2",
    ]


def _service_running(service: str) -> bool:
    state = status()
    for item in state.get("services", []):
        if item.get("Service") == service and str(item.get("State", "")).lower() == "running":
            return True
    return False


def start_crypten_service(max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_running()
    env = warm_env()
    sdir = service_dir()
    for path in [sdir / "stop", sdir / "rank0.ready", sdir / "rank1.ready"]:
        path.unlink(missing_ok=True)
    clear_dir(sdir / "requests")
    clear_dir(sdir / "responses")
    logs_dir = sdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    common_env = _crypten_service_common_env(max_seq_len, layers)
    t0 = time.perf_counter()
    r0 = subprocess.Popen(
        [
            *compose(), "exec", "-T", "crypten-warm-r0", "env", "RANK=0", *common_env,
            "python", "/workspace/docker/crypten_persistent_service.py",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=(logs_dir / "rank0.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.0)
    r1 = subprocess.Popen(
        [
            *compose(), "exec", "-T", "crypten-warm-r1", "env", "RANK=1", *common_env,
            "python", "/workspace/docker/crypten_persistent_service.py",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=(logs_dir / "rank1.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        if (sdir / "rank0.ready").exists() and (sdir / "rank1.ready").exists():
            return {
                "success": True,
                "project": PROJECT,
                "service": "crypten_persistent",
                "max_seq_len": max_seq_len,
                "layers": layers,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "rank0_pid": r0.pid,
                "rank1_pid": r1.pid,
                "service_dir": str(sdir),
            }
        if r0.poll() is not None or r1.poll() is not None:
            break
        time.sleep(0.5)
    return {
        "success": False,
        "project": PROJECT,
        "service": "crypten_persistent",
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "rank0_code": r0.poll(),
        "rank1_code": r1.poll(),
        "service_dir": str(sdir),
    }


def stop_crypten_service() -> dict:
    sdir = service_dir()
    (sdir / "stop").write_text(str(time.time()), encoding="utf-8")
    time.sleep(1.0)
    env = warm_env()
    # Best effort cleanup in case a rank is blocked in Gloo after stop marker.
    for service in ("crypten-warm-r0", "crypten-warm-r1"):
        exec_in(service, ["pkill", "-f", "crypten_persistent_service.py"], env, timeout=30)
    for path in [sdir / "rank0.ready", sdir / "rank1.ready", sdir / "rank0.heartbeat", sdir / "rank1.heartbeat"]:
        path.unlink(missing_ok=True)
    return {"success": True, "project": PROJECT, "service": "crypten_persistent", "service_dir": str(sdir)}


def _mcu_service_common_args(max_seq_len: int = 16, layers: int = 12) -> list[str]:
    cache_id = mcu_model_cache_id(max_seq_len, layers, 16)
    return [
        "--batch", "1",
        "--seq", str(max_seq_len),
        "--hidden", "768",
        "--heads", "12",
        "--ffn", "3072",
        "--layers", str(layers),
        "--state-mode", "chained",
        "--input-mode", "real_io",
        "--scale-bits", "16",
        "--rescale-bits", "16",
        "--rescale-mode", "hp_clear",
        "--service-dir", "/workspace/out/mcu_service",
        "--share-dir", "/workspace/bert_shares",
        "--model-share-dir", f"/workspace/bert_shares/_model_cache/{cache_id}",
        "--out-dir", "/workspace/out/mcu_service/responses",
    ]


def start_mcu_service(max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_running()
    env = warm_env()
    sdir = mcu_service_dir()
    for path in [
        sdir / "stop",
        sdir / "input_exporter.stop",
        sdir / "hp.ready",
        sdir / "p0.ready",
        sdir / "p1.ready",
        sdir / "input_exporter.ready",
        sdir / "hp.heartbeat",
        sdir / "p0.heartbeat",
        sdir / "p1.heartbeat",
        sdir / "input_exporter.heartbeat",
        sdir / "input_exporter.heartbeat.tmp",
    ]:
        path.unlink(missing_ok=True)
    clear_dir(sdir / "input_requests")
    clear_dir(sdir / "input_responses")
    clear_dir(sdir / "requests")
    clear_dir(sdir / "responses")
    logs_dir = sdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    args = _mcu_service_common_args(max_seq_len, layers)
    t0 = time.perf_counter()
    procs = []
    input_proc = subprocess.Popen(
        [sys.executable, str(INPUT_SHARE_SERVICE), "--service-dir", str(sdir)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=(logs_dir / "input_exporter.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    role_specs = [
        ("hp", "mcu-warm-hp", ["service-hp", "--addr", "0.0.0.0:9400", *args]),
        ("p0", "mcu-warm-p0", ["service-p0", "--addr", "mcu-warm-hp:9400", *args]),
        ("p1", "mcu-warm-p1", ["service-p1", "--addr", "mcu-warm-hp:9400", *args]),
    ]
    for role, service, command in role_specs:
        proc = subprocess.Popen(
            [*compose(), "exec", "-T", service, "/workspace/mcu_rust/target/release/bert_session", *command],
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=(logs_dir / f"{role}.log").open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        procs.append((role, proc))
        time.sleep(0.5)
    deadline = time.time() + 120
    while time.time() < deadline:
        role_ready = all(
            (sdir / f"{role}.ready").exists() and heartbeat_fresh(sdir / f"{role}.heartbeat", 120.0)
            for role, _ in procs
        )
        input_ready = (sdir / "input_exporter.ready").exists() and heartbeat_fresh(sdir / "input_exporter.heartbeat", 120.0)
        if role_ready and input_ready:
            for role, _ in procs:
                (sdir / f"{role}.ready").write_text(
                    json.dumps(
                        {
                            "role": role,
                            "state": "ready",
                            "max_seq_len": max_seq_len,
                            "max_layers": layers,
                            "ready_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            return {
                "success": True,
                "project": PROJECT,
                "service": "mcu_persistent",
                "max_seq_len": max_seq_len,
                "layers": layers,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "pids": {role: proc.pid for role, proc in procs} | {"input_exporter": input_proc.pid},
                "service_dir": str(sdir),
                "note": "MCU service v2 keeps wrapper role processes and host input-share exporter alive; inner role TCP still reconnects per request.",
            }
        if any(proc.poll() is not None for _, proc in procs) or input_proc.poll() is not None:
            break
        time.sleep(0.5)
    return {
        "success": False,
        "project": PROJECT,
        "service": "mcu_persistent",
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "codes": {role: proc.poll() for role, proc in procs} | {"input_exporter": input_proc.poll()},
        "service_dir": str(sdir),
    }


def stop_mcu_service() -> dict:
    sdir = mcu_service_dir()
    (sdir / "stop").write_text(str(time.time()), encoding="utf-8")
    (sdir / "input_exporter.stop").write_text(str(time.time()), encoding="utf-8")
    time.sleep(1.0)
    env = warm_env()
    for service in ("mcu-warm-hp", "mcu-warm-p0", "mcu-warm-p1"):
        exec_in(service, ["pkill", "-f", "bert_session"], env, timeout=30)
    for path in [
        sdir / "hp.ready",
        sdir / "p0.ready",
        sdir / "p1.ready",
        sdir / "hp.heartbeat",
        sdir / "p0.heartbeat",
        sdir / "p1.heartbeat",
    ]:
        path.unlink(missing_ok=True)
    return {"success": True, "project": PROJECT, "service": "mcu_persistent", "service_dir": str(sdir)}


def mcu_service_status() -> dict:
    sdir = mcu_service_dir()
    roles = ("hp", "p0", "p1")
    ready_payload = {role: read_json_if_exists(sdir / f"{role}.ready") for role in roles}
    heartbeat_age = {role: file_age_s(sdir / f"{role}.heartbeat") for role in roles}
    heartbeat_age["input_exporter"] = file_age_s(sdir / "input_exporter.heartbeat")
    ready = {
        role: (sdir / f"{role}.ready").exists() and heartbeat_fresh(sdir / f"{role}.heartbeat")
        for role in roles
    }
    ready["input_exporter"] = (sdir / "input_exporter.ready").exists() and heartbeat_fresh(sdir / "input_exporter.heartbeat")
    return {
        "success": all(ready.values()),
        "project": PROJECT,
        "service": "mcu_persistent",
        "ready": ready,
        "config": ready_payload,
        "heartbeat_age_s": heartbeat_age,
        "heartbeat_stale_s": HEARTBEAT_STALE_S,
        "service_dir": str(sdir),
        "note": "MCU service v2 wrapper is persistent; inner protocol connection is still per request.",
    }


def ensure_mcu_service(max_seq_len: int = 16, layers: int = 12) -> None:
    state = mcu_service_status()
    configs = state.get("config", {})
    config_ok = all(
        service_config_matches(configs.get(role, {}), max_seq_len, layers)
        for role in ("hp", "p0", "p1")
    )
    if state.get("success") and not config_ok:
        stop_mcu_service()
        state = mcu_service_status()
    if not state.get("success"):
        started = start_mcu_service(max_seq_len, layers)
        if not started.get("success"):
            raise RuntimeError(f"mcu persistent service failed to start: {started}")


def crypten_service_status() -> dict:
    sdir = service_dir()
    ready = [(sdir / "rank0.ready").exists(), (sdir / "rank1.ready").exists()]
    ready_payload = {f"rank{rank}": read_json_if_exists(sdir / f"rank{rank}.ready") for rank in (0, 1)}
    heartbeats = {}
    for rank in (0, 1):
        path = sdir / f"rank{rank}.heartbeat"
        heartbeats[f"rank{rank}"] = file_age_s(path)
    live = [
        ready[rank] and heartbeat_fresh(sdir / f"rank{rank}.heartbeat")
        for rank in (0, 1)
    ]
    return {
        "success": all(live),
        "project": PROJECT,
        "service": "crypten_persistent",
        "ready": {"rank0": live[0], "rank1": live[1]},
        "ready_marker": {"rank0": ready[0], "rank1": ready[1]},
        "config": ready_payload,
        "heartbeat_age_s": heartbeats,
        "heartbeat_stale_s": HEARTBEAT_STALE_S,
        "service_dir": str(sdir),
    }


def ensure_crypten_service(max_seq_len: int = 16, layers: int = 12) -> None:
    state = crypten_service_status()
    configs = state.get("config", {})
    config_ok = all(
        service_config_matches(configs.get(f"rank{rank}", {}), max_seq_len, layers)
        for rank in (0, 1)
    )
    if state.get("success") and not config_ok:
        stop_crypten_service()
        state = crypten_service_status()
    if not state.get("success"):
        started = start_crypten_service(max_seq_len, layers)
        if not started.get("success"):
            raise RuntimeError(f"crypten persistent service failed to start: {started}")


def export_real_shares(text: str, max_seq_len: int, layers: int, scale_bits: int, target_root: Path, run_id: str) -> Path:
    source_out = Path(tempfile.gettempdir()) / "aegisx_warm_share_exports" / run_id
    if source_out.exists():
        shutil.rmtree(source_out, ignore_errors=True)
    source_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPORT_SHARES),
        "--text",
        text,
        "--max-seq-len",
        str(max_seq_len),
        "--layers",
        str(layers),
        "--scale-bits",
        str(scale_bits),
    ]
    completed = run(cmd, os.environ.copy(), timeout=600, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"share export failed:\n{completed.stdout}\n{completed.stderr}")
    match = re.search(r"\[bert-shares\] wrote (.+?manifest\.json)", completed.stdout)
    if not match:
        raise RuntimeError(f"could not parse share export path:\n{completed.stdout}\n{completed.stderr}")
    source = Path(match.group(1)).parent
    target = target_root / run_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    if source != target and source.exists():
        shutil.rmtree(source, ignore_errors=True)
    return target


def _run_share_export(
    text: str,
    max_seq_len: int,
    layers: int,
    scale_bits: int,
    mode_flag: str | None = None,
    timeout: int = 600,
) -> Path:
    cmd = [
        sys.executable,
        str(EXPORT_SHARES),
        "--text",
        text,
        "--max-seq-len",
        str(max_seq_len),
        "--layers",
        str(layers),
        "--scale-bits",
        str(scale_bits),
    ]
    if mode_flag:
        cmd.append(mode_flag)
    completed = run(cmd, os.environ.copy(), timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"share export failed:\n{completed.stdout}\n{completed.stderr}")
    match = re.search(r"\[bert-shares\] wrote (.+?manifest\.json)", completed.stdout)
    if not match:
        raise RuntimeError(f"could not parse share export path:\n{completed.stdout}\n{completed.stderr}")
    return Path(match.group(1)).parent


def ensure_mcu_model_share_cache(max_seq_len: int, layers: int, scale_bits: int = 16) -> tuple[Path, float, bool]:
    target = mcu_model_cache_dir(max_seq_len, layers, scale_bits)
    manifest = target / "manifest.json"
    if manifest.exists():
        return target, 0.0, False
    t0 = time.perf_counter()
    source = _run_share_export(DEFAULT_CACHE_TEXT, max_seq_len, layers, scale_bits, "--static-only", timeout=900)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    if source != target and source.exists():
        shutil.rmtree(source, ignore_errors=True)
    return target, time.perf_counter() - t0, True


def export_input_shares(text: str, max_seq_len: int, layers: int, scale_bits: int, target_root: Path, run_id: str) -> tuple[Path, float]:
    t0 = time.perf_counter()
    sdir = mcu_service_dir()
    if (sdir / "input_exporter.ready").exists():
        target = target_root / run_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        req_path = sdir / "input_requests" / f"{run_id}.json"
        done_path = sdir / "input_responses" / f"{run_id}.done"
        error_path = sdir / "input_responses" / f"{run_id}.error"
        done_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        req_path.write_text(
            json.dumps(
                {
                    "text": text,
                    "max_seq_len": max_seq_len,
                    "scale_bits": scale_bits,
                    "out_dir": str(target),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if error_path.exists():
                raise RuntimeError(f"input share service failed: {error_path.read_text(encoding='utf-8')}")
            if done_path.exists():
                return target, time.perf_counter() - t0
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for input share service response {run_id}")
    source = _run_share_export(text, max_seq_len, layers, scale_bits, "--input-only", timeout=600)
    target = target_root / run_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    if source != target and source.exists():
        shutil.rmtree(source, ignore_errors=True)
    return target, time.perf_counter() - t0


def exec_in(service: str, command: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    return run([*compose(), "exec", "-T", service, *command], env, timeout=timeout, check=False)


def parse_timing(role: str, text: str) -> dict:
    match = re.search(rf"\[{re.escape(role)}\] timing (?P<body>[^\n]+)", text)
    row = {"role": role}
    if not match:
        return row
    for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", match.group("body")):
        row[key] = value
    for key, value in re.findall(
        r"(send_msgs|recv_msgs|send_bytes|recv_bytes|matmul_calls|cpu_matmul_calls|cuda_matmul_calls|fused_party_calls|fused_hp_calls|cuda_fallbacks)=([0-9]+)",
        match.group("body"),
    ):
        row[key] = int(value)
    return row


def parse_party_out(path: Path) -> dict:
    out: dict[str, object] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"prediction", "layers"}:
            out[key] = int(value)
        elif key == "predictions":
            out[key] = [int(x) for x in value.split(",") if x]
        elif key == "prediction_labels":
            out[key] = [x for x in value.split(",") if x]
        elif key in {"logits", "probabilities"}:
            out[key] = [float(x) for x in value.split(",") if x]
        else:
            out[key] = value
    return out


def infer_mcu(text: str, max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_running()
    env = warm_env()
    out_root, share_root = warm_paths()
    run_id = datetime.now().strftime("mcu_%Y%m%d_%H%M%S_%f")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    share_dir = export_real_shares(text, max_seq_len, layers, 16, share_root, run_id)
    container_out = f"/workspace/out/{run_id}"
    common = [
        "--batch", "1",
        "--seq", str(max_seq_len),
        "--hidden", "768",
        "--heads", "12",
        "--ffn", "3072",
        "--layers", str(layers),
        "--state-mode", "chained",
        "--input-mode", "real_io",
        "--scale-bits", "16",
        "--rescale-bits", "16",
        "--rescale-mode", "hp_clear",
    ]
    t0 = time.perf_counter()
    hp = subprocess.Popen(
        [*compose(), "exec", "-T", "mcu-warm-hp", "/workspace/mcu_rust/target/release/bert_session", "hp", "--addr", "0.0.0.0:9400", *common],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    p0 = subprocess.Popen(
        [
            *compose(), "exec", "-T", "mcu-warm-p0", "/workspace/mcu_rust/target/release/bert_session", "p0",
            "--addr", "mcu-warm-hp:9400", *common, "--share-dir", f"/workspace/bert_shares/{run_id}/p0", "--out", f"{container_out}/p0.out",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p1 = subprocess.Popen(
        [
            *compose(), "exec", "-T", "mcu-warm-p1", "/workspace/mcu_rust/target/release/bert_session", "p1",
            "--addr", "mcu-warm-hp:9400", *common, "--share-dir", f"/workspace/bert_shares/{run_id}/p1", "--out", f"{container_out}/p1.out",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        p0_out, p0_err = p0.communicate(timeout=600)
        p1_out, p1_err = p1.communicate(timeout=600)
        hp_out, hp_err = hp.communicate(timeout=120)
    finally:
        for proc in (p0, p1, hp):
            if proc.poll() is None:
                proc.kill()
    elapsed = time.perf_counter() - t0
    (out_dir / "p0.log").write_text(p0_out + p0_err, encoding="utf-8")
    (out_dir / "p1.log").write_text(p1_out + p1_err, encoding="utf-8")
    (out_dir / "hp.log").write_text(hp_out + hp_err, encoding="utf-8")
    if p0.returncode != 0 or p1.returncode != 0 or hp.returncode != 0:
        raise RuntimeError(f"warm MCU failed: hp={hp.returncode}, p0={p0.returncode}, p1={p1.returncode}")
    timing_rows = [parse_timing("hp", hp_out + hp_err), parse_timing("p0", p0_out + p0_err), parse_timing("p1", p1_out + p1_err)]
    write_csv(out_dir / "role_timing.csv", timing_rows)
    party_result = parse_party_out(out_dir / "p0.out")
    critical = max((float(row.get("total_s", 0.0)) for row in timing_rows), default=elapsed)
    summary = {
        "mode": "mcu_warm_docker_exec",
        "status": "ok",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "critical_role_total_s": critical,
        "prediction": party_result.get("prediction", ""),
        "prediction_label": party_result.get("prediction_label", ""),
        "probabilities": party_result.get("probabilities", []),
        "output_dir": str(out_dir),
        "note": "Warm Docker v1 keeps containers alive and executes one bert_session request per API call. The Rust p0/p1/hp process still exits after each request.",
    }
    (out_dir / "result.json").write_text(json.dumps({"summary": summary, "result": party_result, "role_timing": timing_rows}, indent=2), encoding="utf-8")
    print(f"[warm-mcu] wrote {out_dir / 'result.json'}")
    return {"summary": summary, "result": party_result, "role_timing": timing_rows}


def infer_mcu_service(text: str, max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_mcu_service(max_seq_len, layers)
    out_root, share_root = warm_paths()
    sdir = mcu_service_dir()
    run_id = datetime.now().strftime("mcu_service_%Y%m%d_%H%M%S_%f")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache, model_cache_s, model_cache_created = ensure_mcu_model_share_cache(max_seq_len, layers, 16)
    _, input_export_s = export_input_shares(text, max_seq_len, layers, 16, share_root, run_id)
    req_path = sdir / "requests" / f"{run_id}.req"
    resp_dir = sdir / "responses" / run_id
    resp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    req_path.write_text(f"text={text}\n", encoding="utf-8")
    deadline = time.time() + SERVICE_REQUEST_TIMEOUT_S
    done_paths = [resp_dir / "hp.done", resp_dir / "p0.done", resp_dir / "p1.done"]
    error_paths = [resp_dir / "hp.error", resp_dir / "p0.error", resp_dir / "p1.error"]
    while time.time() < deadline:
        for err in error_paths:
            if err.exists():
                raise RuntimeError(f"MCU service role failed: {err.read_text(encoding='utf-8')}")
        state = mcu_service_status()
        if not state.get("success"):
            raise RuntimeError(f"MCU service became unavailable while processing {run_id}: {state}")
        if all(path.exists() for path in done_paths):
            elapsed = time.perf_counter() - t0
            p0_out = resp_dir / "p0.out"
            p1_out = resp_dir / "p1.out"
            if p0_out.exists():
                shutil.copy2(p0_out, out_dir / "p0.out")
            if p1_out.exists():
                shutil.copy2(p1_out, out_dir / "p1.out")
            party_result = parse_party_out(p0_out)
            role_rows = []
            for role in ("hp", "p0", "p1"):
                done = resp_dir / f"{role}.done"
                row = {"role": role}
                for line in done.read_text(encoding="utf-8").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        row[key] = value
                role_rows.append(row)
            write_csv(out_dir / "role_timing.csv", role_rows)
            summary = {
                "mode": "mcu_persistent_service",
                "status": "ok",
                "run_id": run_id,
                "elapsed_seconds": elapsed,
                "prediction": party_result.get("prediction", ""),
                "prediction_label": party_result.get("prediction_label", ""),
                "probabilities": party_result.get("probabilities", []),
                "output_dir": str(out_dir),
                "model_share_cache": str(model_cache),
                "model_share_cache_created": model_cache_created,
                "model_share_cache_s": model_cache_s,
                "input_share_export_s": input_export_s,
                "note": "MCU persistent v2 wrapper keeps role service processes alive. Static model shares are cached; input shares, inner TCP reconnect, and request-local reads still happen per request.",
            }
            (out_dir / "result.json").write_text(
                json.dumps({"summary": summary, "result": party_result, "role_timing": role_rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[service-mcu] wrote {out_dir / 'result.json'}")
            return {"summary": summary, "result": party_result, "role_timing": role_rows}
        time.sleep(0.2)
    raise TimeoutError(f"timed out after {SERVICE_REQUEST_TIMEOUT_S:.1f}s waiting for MCU service response {run_id}")


def infer_crypten(text: str, max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_running()
    env = warm_env()
    out_root, _ = warm_paths()
    run_id = datetime.now().strftime("crypten_%Y%m%d_%H%M%S_%f")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.json"
    samples_path.write_text(json.dumps({"samples": [{"text": text}]}, ensure_ascii=False), encoding="utf-8")
    container_out = f"/workspace/out/{run_id}"
    common_env = [
        "CRYPTEN_OP=bert_full",
        "CRYPTEN_REPEAT=1",
        f"CRYPTEN_MAX_SEQ_LEN={max_seq_len}",
        f"CRYPTEN_MAX_LAYERS={layers}",
        "CRYPTEN_BERT_NONLINEAR=native",
        f"CRYPTEN_INPUT_JSON={container_out}/samples.json",
        f"CRYPTEN_OUT_DIR={container_out}",
        "RENDEZVOUS=tcp://crypten-warm-r0:29500",
        "DISTRIBUTED_BACKEND=gloo",
        "WORLD_SIZE=2",
    ]
    t0 = time.perf_counter()
    r0 = subprocess.Popen(
        [*compose(), "exec", "-T", "crypten-warm-r0", "env", "RANK=0", *common_env, "python", "/workspace/docker/crypten_rank_bench.py"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    r1 = subprocess.Popen(
        [*compose(), "exec", "-T", "crypten-warm-r1", "env", "RANK=1", *common_env, "python", "/workspace/docker/crypten_rank_bench.py"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        r1_out, r1_err = r1.communicate(timeout=1200)
        r0_out, r0_err = r0.communicate(timeout=1200)
    finally:
        for proc in (r0, r1):
            if proc.poll() is None:
                proc.kill()
    elapsed = time.perf_counter() - t0
    (out_dir / "rank0.log").write_text(r0_out + r0_err, encoding="utf-8")
    (out_dir / "rank1.log").write_text(r1_out + r1_err, encoding="utf-8")
    if r0.returncode != 0 or r1.returncode != 0:
        raise RuntimeError(f"warm CrypTen failed: r0={r0.returncode}, r1={r1.returncode}")
    rank0 = json.loads((out_dir / "rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank1.json").read_text(encoding="utf-8"))
    prediction = rank0.get("predictions", [{}])[0]
    summary = {
        "mode": "crypten_warm_docker_exec",
        "status": "ok",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "prediction": prediction.get("prediction", ""),
        "label": prediction.get("label", ""),
        "probabilities": prediction.get("probabilities", []),
        "latency_s": prediction.get("latency_s", elapsed),
        "output_dir": str(out_dir),
        "note": "Warm Docker v1 keeps containers alive and executes one CrypTen rank process per API call. The rank process still exits after each request.",
    }
    (out_dir / "result.json").write_text(json.dumps({"summary": summary, "crypten_rank0": rank0, "crypten_rank1": rank1}, indent=2), encoding="utf-8")
    print(f"[warm-crypten] wrote {out_dir / 'result.json'}")
    return {"summary": summary, "crypten_rank0": rank0, "crypten_rank1": rank1}


def infer_crypten_service(text: str, max_seq_len: int = 16, layers: int = 12) -> dict:
    ensure_crypten_service(max_seq_len, layers)
    sdir = service_dir()
    out_root, _ = warm_paths()
    run_id = datetime.now().strftime("crypten_service_%Y%m%d_%H%M%S_%f")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    req = {
        "request_id": run_id,
        "text": text,
        "max_seq_len": max_seq_len,
        "layers": layers,
        "created_at": time.time(),
    }
    req_path = sdir / "requests" / f"{run_id}.json"
    resp_path = sdir / "responses" / f"{run_id}.json"
    t0 = time.perf_counter()
    tmp = req_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(req_path)
    deadline = time.time() + SERVICE_REQUEST_TIMEOUT_S
    while time.time() < deadline:
        if resp_path.exists():
            payload = json.loads(resp_path.read_text(encoding="utf-8"))
            elapsed = time.perf_counter() - t0
            prediction = payload.get("predictions", [{}])[0]
            summary = {
                "mode": "crypten_persistent_service",
                "status": "ok",
                "run_id": run_id,
                "elapsed_seconds": elapsed,
                "prediction": prediction.get("prediction", ""),
                "label": prediction.get("label", ""),
                "probabilities": prediction.get("probabilities", []),
                "latency_s": prediction.get("latency_s", payload.get("elapsed_s", elapsed)),
                "service_elapsed_s": payload.get("elapsed_s", elapsed),
                "output_dir": str(out_dir),
                "note": "CrypTen persistent v2 keeps rank processes and BERT model/tokenizer alive across requests.",
            }
            (out_dir / "result.json").write_text(
                json.dumps({"summary": summary, "crypten_rank0": payload}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[service-crypten] wrote {out_dir / 'result.json'}")
            return {"summary": summary, "crypten_rank0": payload}
        time.sleep(0.1)
    raise TimeoutError(f"timed out after {SERVICE_REQUEST_TIMEOUT_S:.1f}s waiting for CrypTen service response {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("status")
    svc_start = sub.add_parser("start-crypten-service")
    svc_start.add_argument("--max-seq-len", type=int, default=16)
    svc_start.add_argument("--layers", type=int, default=12)
    sub.add_parser("stop-crypten-service")
    sub.add_parser("crypten-service-status")
    mcu_start = sub.add_parser("start-mcu-service")
    mcu_start.add_argument("--max-seq-len", type=int, default=16)
    mcu_start.add_argument("--layers", type=int, default=12)
    sub.add_parser("stop-mcu-service")
    sub.add_parser("mcu-service-status")
    for name in ("infer-mcu", "infer-crypten"):
        p = sub.add_parser(name)
        p.add_argument("--text", default="This movie is wonderful and heartwarming.")
        p.add_argument("--max-seq-len", type=int, default=16)
        p.add_argument("--layers", type=int, default=12)
    p = sub.add_parser("infer-crypten-service")
    p.add_argument("--text", default="This movie is wonderful and heartwarming.")
    p.add_argument("--max-seq-len", type=int, default=16)
    p.add_argument("--layers", type=int, default=12)
    p = sub.add_parser("infer-mcu-service")
    p.add_argument("--text", default="This movie is wonderful and heartwarming.")
    p.add_argument("--max-seq-len", type=int, default=16)
    p.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    if args.command == "start":
        print(json.dumps(start(), ensure_ascii=False, indent=2))
    elif args.command == "stop":
        print(json.dumps(stop(), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "start-crypten-service":
        print(json.dumps(start_crypten_service(args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    elif args.command == "stop-crypten-service":
        print(json.dumps(stop_crypten_service(), ensure_ascii=False, indent=2))
    elif args.command == "crypten-service-status":
        print(json.dumps(crypten_service_status(), ensure_ascii=False, indent=2))
    elif args.command == "start-mcu-service":
        print(json.dumps(start_mcu_service(args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    elif args.command == "stop-mcu-service":
        print(json.dumps(stop_mcu_service(), ensure_ascii=False, indent=2))
    elif args.command == "mcu-service-status":
        print(json.dumps(mcu_service_status(), ensure_ascii=False, indent=2))
    elif args.command == "infer-mcu":
        print(json.dumps(infer_mcu(args.text, args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    elif args.command == "infer-crypten":
        print(json.dumps(infer_crypten(args.text, args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    elif args.command == "infer-crypten-service":
        print(json.dumps(infer_crypten_service(args.text, args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    elif args.command == "infer-mcu-service":
        print(json.dumps(infer_mcu_service(args.text, args.max_seq_len, args.layers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
