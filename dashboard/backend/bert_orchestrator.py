from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP = ROOT / ".local" / "dashboard"
HISTORY_PATH = LOCAL_TMP / "bert_history.jsonl"
LABELS = ["negative", "positive"]
WARM_SERVICE = ROOT / "experiments" / "docker_bert_full" / "warm_docker_service.py"

_last_status: dict[str, Any] = {"state": "idle", "updated_at": None}
_last_logs: list[dict[str, Any]] = []


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _role_log(sender: str, receiver: str, detail: str, bytes_: int = 0) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "sender": sender,
        "receiver": receiver,
        "type": "Docker",
        "detail": detail,
        "bytes": int(bytes_),
    }


def _set_status(state: str, **extra: Any) -> None:
    _last_status.clear()
    _last_status.update({"state": state, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra})


def _set_logs(logs: list[dict[str, Any]]) -> None:
    _last_logs.clear()
    _last_logs.extend(logs)


def _record_history(row: dict[str, Any]) -> None:
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _distribution(probs: list[float]) -> list[dict[str, Any]]:
    rows = [{"label": label, "prob": round(float(prob) * 100.0, 1)} for label, prob in zip(LABELS, probs)]
    return sorted(rows, key=lambda x: -x["prob"])


def _attention_from_text(text: str) -> list[dict[str, Any]]:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    base = 1.0 / len(words)
    return [{"word": word, "weight": round(base, 4)} for word in words[:32]]


def _normalize(
    *,
    text: str,
    mode: str,
    launch: str,
    method: str,
    probabilities: list[float],
    elapsed_seconds: float,
    logs: list[dict[str, Any]],
    artifact_dir: str | None = None,
    latency: dict[str, Any] | None = None,
    security_note: str | None = None,
) -> dict[str, Any]:
    prediction = int(max(range(len(probabilities)), key=lambda i: probabilities[i])) if probabilities else 0
    label = LABELS[prediction]
    result = {
        "success": True,
        "input": text,
        "mode": mode,
        "launch": launch,
        "method": method,
        "label": label,
        "top_prediction": label,
        "prediction": prediction,
        "confidence": round(float(probabilities[prediction]) * 100.0, 1) if probabilities else 0.0,
        "distribution": _distribution(probabilities),
        "probabilities": probabilities,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "latency": latency or {"total_s": round(float(elapsed_seconds), 6)},
        "attention": _attention_from_text(text),
        "logs": logs,
        "comm_rounds": len(logs),
        "artifact_dir": artifact_dir,
        "security_note": security_note or "",
    }
    _set_logs(logs)
    _record_history(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "launch": launch,
            "label": label,
            "confidence": result["confidence"],
            "elapsed_seconds": result["elapsed_seconds"],
            "artifact_dir": artifact_dir,
        }
    )
    return result


def infer_process(text: str, mode: str, max_seq_len: int) -> dict[str, Any]:
    mapped_mode = "mcu_rust" if mode in {"mcu", "mcu_rust"} else mode
    if mapped_mode not in {"plaintext", "crypten", "mcu_rust"}:
        raise ValueError(f"invalid mode: {mode}")
    _set_status("running", mode=mapped_mode, launch="process")
    from bert_inference import get_engine

    out = get_engine().classify(text, mode=mapped_mode, max_seq_len=max_seq_len)
    probs = [float(x) for x in out.get("probabilities", [])]
    logs = [
        _role_log("P0", "P1", "host process plaintext baseline" if mapped_mode == "plaintext" else "host process BERT request"),
    ]
    if mapped_mode == "crypten":
        logs = [
            _role_log("R0", "R1", "CrypTen process-mode tensor exchange"),
            _role_log("R1", "R0", "CrypTen process-mode result synchronization"),
        ]
    elif mapped_mode == "mcu_rust":
        logs = [
            _role_log("P0", "HP", "MCU-Rust process-mode input share path"),
            _role_log("P1", "HP", "MCU-Rust process-mode model share path"),
            _role_log("HP", "P0", "MCU-Rust process-mode assisted nonlinear path"),
            _role_log("HP", "P1", "MCU-Rust process-mode assisted nonlinear path"),
        ]
    result = _normalize(
        text=text,
        mode=mapped_mode,
        launch="process",
        method=f"{out.get('method', mapped_mode)} | process",
        probabilities=probs,
        elapsed_seconds=float(out.get("elapsed_seconds", 0.0)),
        logs=logs,
        latency={"total_s": float(out.get("elapsed_seconds", 0.0)), "startup_s": 0.0},
        security_note="Process mode is useful for quick comparison; Docker mode is the real communication demonstration.",
    )
    _set_status("complete", mode=mapped_mode, launch="process", artifact_dir=None)
    return result


def infer_plaintext_docker_baseline(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="plaintext", launch="docker")
    from bert_inference import get_engine

    out = get_engine().classify(text, mode="plaintext", max_seq_len=max_seq_len)
    result = _normalize(
        text=text,
        mode="plaintext",
        launch="docker",
        method=f"{out.get('method', 'Plaintext BERT')} | host baseline for Docker comparison",
        probabilities=[float(x) for x in out.get("probabilities", [])],
        elapsed_seconds=float(out.get("elapsed_seconds", 0.0)),
        logs=[_role_log("HOST", "HOST", "Plaintext host baseline used for Docker comparison")],
        latency={"total_s": float(out.get("elapsed_seconds", 0.0)), "startup_s": 0.0},
        security_note="Plaintext has no multi-party Docker role; it is measured as the host baseline.",
    )
    _set_status("complete", mode="plaintext", launch="docker", artifact_dir=None)
    return result


def _extract_output_dir(stdout: str, prefix: str) -> Path:
    match = re.search(rf"\[{re.escape(prefix)}\] wrote (.+?summary\.csv)", stdout)
    if not match:
        raise RuntimeError(f"could not parse output directory from stdout:\n{stdout}")
    return Path(match.group(1)).resolve().parent


def infer_crypten_docker(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="crypten", launch="docker")
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    sample_path = LOCAL_TMP / f"crypten_sample_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    sample_path.write_text(json.dumps({"samples": [text]}, ensure_ascii=False), encoding="utf-8")
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "docker_bert_full" / "run_docker_bert_comparison.py"),
        "--samples",
        "1",
        "--repeat",
        "1",
        "--max-seq-len",
        str(max_seq_len),
        "--max-layers",
        "12",
        "--crypten-nonlinear",
        "native",
        "--skip-build",
        "--samples-json",
        str(sample_path),
    ]
    completed = _run(cmd, timeout=2400)
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        _set_status("failed", mode="crypten", launch="docker")
        raise RuntimeError(f"CrypTen Docker inference failed:\n{completed.stdout}\n{completed.stderr}")
    out_dir = _extract_output_dir(completed.stdout, "bert-docker")
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    pred = payload["crypten_rank0"]["predictions"][0]
    probs = [float(x) for x in pred["probabilities"]]
    logs = [
        _role_log("R0", "R1", "CrypTen Docker rank0 started over Gloo/TCP"),
        _role_log("R1", "R0", "CrypTen Docker rank1 completed secure BERT request"),
    ]
    result = _normalize(
        text=text,
        mode="crypten",
        launch="docker",
        method="CrypTen Docker 12L BERT native nonlinear",
        probabilities=probs,
        elapsed_seconds=float(pred.get("latency_s", elapsed)),
        logs=logs,
        artifact_dir=str(out_dir),
        latency={"total_s": float(elapsed), "protocol_s": float(pred.get("latency_s", elapsed))},
        security_note="Docker mode starts two CrypTen ranks over Gloo/TCP for the BERT request.",
    )
    _set_status("complete", mode="crypten", launch="docker", artifact_dir=str(out_dir))
    return result


def infer_mcu_docker(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="mcu_rust", launch="docker")
    project = f"aegisxmcu{datetime.now().strftime('%H%M%S')}"
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "docker_bert_full" / "run_mcu_bert_session_docker.py"),
        "--image",
        os.environ.get("MCU_DASHBOARD_IMAGE", "phantomshield-mcu:bert-session-fixed"),
        "--batch",
        "1",
        "--seq",
        str(max_seq_len),
        "--hidden",
        "768",
        "--heads",
        "12",
        "--ffn",
        "3072",
        "--layers",
        "12",
        "--project",
        project,
        "--state-mode",
        "chained",
        "--input-mode",
        "real_io",
        "--text",
        text,
        "--scale-bits",
        "16",
        "--rescale-mode",
        "hp_clear",
    ]
    completed = _run(cmd, timeout=2400)
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        _set_status("failed", mode="mcu_rust", launch="docker")
        raise RuntimeError(f"MCU Docker inference failed:\n{completed.stdout}\n{completed.stderr}")
    out_dir = _extract_output_dir(completed.stdout, "mcu-bert-docker")
    payload = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    role_timing = payload.get("role_timing", [])
    probs = [float(x) for x in payload.get("result", {}).get("probabilities", [])[:2]]
    critical = max((float(row.get("total_s", 0.0)) for row in role_timing), default=elapsed)
    logs = []
    for row in role_timing:
        role = str(row.get("role", "")).upper()
        send_bytes = int(row.get("send_bytes", 0) or 0)
        recv_bytes = int(row.get("recv_bytes", 0) or 0)
        detail = (
            f"MCU Docker role total={row.get('total_s', '?')}s "
            f"send={row.get('send_msgs', 0)} recv={row.get('recv_msgs', 0)}"
        )
        if role == "HP":
            logs.append(_role_log("HP", "P0/P1", detail, send_bytes + recv_bytes))
        else:
            logs.append(_role_log(role, "HP", detail, send_bytes + recv_bytes))
    if not logs:
        logs = [
            _role_log("P0", "HP", "MCU Docker p0 role completed"),
            _role_log("P1", "HP", "MCU Docker p1 role completed"),
            _role_log("HP", "P0/P1", "MCU Docker helper role completed"),
        ]
    result = _normalize(
        text=text,
        mode="mcu_rust",
        launch="docker",
        method="MCU Docker p0/p1/hp 12L BERT numerical prototype",
        probabilities=probs,
        elapsed_seconds=critical,
        logs=logs,
        artifact_dir=str(out_dir),
        latency={"total_s": float(elapsed), "critical_role_s": float(critical)},
        security_note="MCU Docker uses real p0/p1/hp TCP, but still contains HP-clear numerical bridges for rescale, nonlinear feedback, LayerNorm, and classifier output.",
    )
    _set_status("complete", mode="mcu_rust", launch="docker", artifact_dir=str(out_dir))
    return result


def _run_warm_service(args: list[str], timeout: int = 2400) -> dict[str, Any]:
    completed = _run([sys.executable, str(WARM_SERVICE), *args], timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"warm Docker service command failed:\n{completed.stdout}\n{completed.stderr}")
    text = completed.stdout.strip()
    match = re.search(r"(\{[\s\S]*\})\s*$", text)
    if not match:
        raise RuntimeError(f"could not parse warm service JSON:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(match.group(1))


def infer_crypten_docker_warm(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="crypten", launch="docker_warm")
    t0 = time.perf_counter()
    payload = _run_warm_service(["infer-crypten", "--text", text, "--max-seq-len", str(max_seq_len), "--layers", "12"])
    elapsed = time.perf_counter() - t0
    summary = payload.get("summary", {})
    probs = [float(x) for x in summary.get("probabilities", [])[:2]]
    out_dir = summary.get("output_dir")
    result = _normalize(
        text=text,
        mode="crypten",
        launch="docker_warm",
        method="CrypTen warm Docker service v1 (containers stay up; rank process per request)",
        probabilities=probs,
        elapsed_seconds=float(summary.get("latency_s", elapsed)),
        logs=[
            _role_log("API", "crypten-warm-r0/r1", "Warm containers reused; executed CrypTen ranks for this request"),
            _role_log("R0", "R1", "CrypTen Gloo/TCP request completed"),
        ],
        artifact_dir=str(out_dir) if out_dir else None,
        latency={
            "total_s": float(elapsed),
            "protocol_s": float(summary.get("latency_s", elapsed)),
            "container_start_s": 0.0,
            "warm_service_v": 1,
        },
        security_note="Warm Docker v1 removes container cold start, but CrypTen rank processes still execute once per request.",
    )
    _set_status("complete", mode="crypten", launch="docker_warm", artifact_dir=str(out_dir) if out_dir else None)
    return result


def infer_crypten_docker_service(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="crypten", launch="docker_service")
    t0 = time.perf_counter()
    payload = _run_warm_service(["infer-crypten-service", "--text", text, "--max-seq-len", str(max_seq_len), "--layers", "12"], timeout=150)
    elapsed = time.perf_counter() - t0
    summary = payload.get("summary", {})
    probs = [float(x) for x in summary.get("probabilities", [])[:2]]
    out_dir = summary.get("output_dir")
    result = _normalize(
        text=text,
        mode="crypten",
        launch="docker_service",
        method="CrypTen persistent Docker service v2 (rank processes and model stay loaded)",
        probabilities=probs,
        elapsed_seconds=float(summary.get("latency_s", elapsed)),
        logs=[
            _role_log("API", "crypten-service", "Submitted request JSON to persistent CrypTen rank service"),
            _role_log("R0", "R1", "Persistent CrypTen rank service completed BERT request"),
        ],
        artifact_dir=str(out_dir) if out_dir else None,
        latency={
            "total_s": float(elapsed),
            "protocol_s": float(summary.get("latency_s", elapsed)),
            "service_elapsed_s": float(summary.get("service_elapsed_s", elapsed)),
            "container_start_s": 0.0,
            "model_load_s": 0.0,
            "warm_service_v": 2,
        },
        security_note="CrypTen persistent v2 keeps Docker containers, rank processes, and model/tokenizer loaded across requests.",
    )
    _set_status("complete", mode="crypten", launch="docker_service", artifact_dir=str(out_dir) if out_dir else None)
    return result


def infer_mcu_docker_warm(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="mcu_rust", launch="docker_warm")
    t0 = time.perf_counter()
    payload = _run_warm_service(["infer-mcu", "--text", text, "--max-seq-len", str(max_seq_len), "--layers", "12"])
    elapsed = time.perf_counter() - t0
    summary = payload.get("summary", {})
    role_timing = payload.get("role_timing", [])
    probs = [float(x) for x in summary.get("probabilities", [])[:2]]
    out_dir = summary.get("output_dir")
    logs = []
    for row in role_timing:
        role = str(row.get("role", "")).upper()
        send_bytes = int(row.get("send_bytes", 0) or 0)
        recv_bytes = int(row.get("recv_bytes", 0) or 0)
        detail = (
            f"warm MCU role total={row.get('total_s', '?')}s "
            f"send={row.get('send_msgs', 0)} recv={row.get('recv_msgs', 0)}"
        )
        logs.append(_role_log(role or "MCU", "HP" if role != "HP" else "P0/P1", detail, send_bytes + recv_bytes))
    if not logs:
        logs = [_role_log("API", "mcu-warm-p0/p1/hp", "Warm containers reused; executed bert_session request")]
    critical = float(summary.get("critical_role_total_s", elapsed) or elapsed)
    result = _normalize(
        text=text,
        mode="mcu_rust",
        launch="docker_warm",
        method="MCU warm Docker service v1 p0/p1/hp numerical prototype",
        probabilities=probs,
        elapsed_seconds=critical,
        logs=logs,
        artifact_dir=str(out_dir) if out_dir else None,
        latency={
            "total_s": float(elapsed),
            "critical_role_s": critical,
            "container_start_s": 0.0,
            "warm_service_v": 1,
        },
        security_note="Warm Docker v1 keeps p0/p1/hp containers alive, but the Rust bert_session process still exits after each request. HP-clear numerical bridges remain.",
    )
    _set_status("complete", mode="mcu_rust", launch="docker_warm", artifact_dir=str(out_dir) if out_dir else None)
    return result


def infer_mcu_docker_service(text: str, max_seq_len: int) -> dict[str, Any]:
    _set_status("running", mode="mcu_rust", launch="docker_service")
    t0 = time.perf_counter()
    payload = _run_warm_service(["infer-mcu-service", "--text", text, "--max-seq-len", str(max_seq_len), "--layers", "12"], timeout=150)
    elapsed = time.perf_counter() - t0
    summary = payload.get("summary", {})
    probs = [float(x) for x in summary.get("probabilities", [])[:2]]
    out_dir = summary.get("output_dir")
    result = _normalize(
        text=text,
        mode="mcu_rust",
        launch="docker_service",
        method="MCU persistent Docker service v2 wrapper p0/p1/hp numerical prototype",
        probabilities=probs,
        elapsed_seconds=float(summary.get("elapsed_seconds", elapsed)),
        logs=[
            _role_log("API", "mcu-service", "Submitted request to persistent MCU role service wrapper"),
            _role_log("P0/P1/HP", "API", "MCU service request completed"),
        ],
        artifact_dir=str(out_dir) if out_dir else None,
        latency={
            "total_s": float(elapsed),
            "service_elapsed_s": float(summary.get("elapsed_seconds", elapsed)),
            "container_start_s": 0.0,
            "warm_service_v": 2,
        },
        security_note="MCU service v2 wrapper keeps role service processes alive, but the inner bert_session protocol still reconnects and rereads shares per request. HP-clear bridges remain.",
    )
    _set_status("complete", mode="mcu_rust", launch="docker_service", artifact_dir=str(out_dir) if out_dir else None)
    return result


def infer(text: str, mode: str, launch: str, max_seq_len: int) -> dict[str, Any]:
    normalized_mode = "mcu_rust" if mode in {"mcu", "mcu_rust"} else mode
    normalized_launch = launch or "process"
    if normalized_launch == "process":
        return infer_process(text, normalized_mode, max_seq_len)
    if normalized_launch not in {"docker", "docker_warm", "docker_service"}:
        raise ValueError(f"invalid launch: {launch}")
    if normalized_mode == "plaintext":
        if normalized_launch in {"docker_warm", "docker_service"}:
            out = infer_plaintext_docker_baseline(text, max_seq_len)
            out["launch"] = normalized_launch
            out["method"] = f"{out['method']} | {normalized_launch} comparison"
            out["latency"]["container_start_s"] = 0.0
            return out
        return infer_plaintext_docker_baseline(text, max_seq_len)
    if normalized_mode == "crypten":
        if normalized_launch == "docker_service":
            return infer_crypten_docker_service(text, max_seq_len)
        if normalized_launch == "docker_warm":
            return infer_crypten_docker_warm(text, max_seq_len)
        return infer_crypten_docker(text, max_seq_len)
    if normalized_mode == "mcu_rust":
        if normalized_launch == "docker_service":
            return infer_mcu_docker_service(text, max_seq_len)
        if normalized_launch == "docker_warm":
            return infer_mcu_docker_warm(text, max_seq_len)
        return infer_mcu_docker(text, max_seq_len)
    raise ValueError(f"invalid mode: {mode}")


def compare(text: str, launch: str, max_seq_len: int, modes: list[str] | None = None) -> dict[str, Any]:
    selected_modes = modes or ["plaintext", "crypten", "mcu_rust"]
    results = []
    all_logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    _set_status("running", mode="compare", launch=launch)
    for mode in selected_modes:
        try:
            item = infer(text, mode, launch, max_seq_len)
            results.append(item)
            all_logs.extend(item.get("logs", []))
        except Exception as exc:
            normalized_mode = "mcu_rust" if mode in {"mcu", "mcu_rust"} else mode
            error_item = {
                "success": False,
                "input": text,
                "mode": normalized_mode,
                "launch": launch,
                "method": f"{normalized_mode} | {launch}",
                "label": "",
                "top_prediction": "",
                "prediction": None,
                "confidence": 0.0,
                "distribution": [],
                "probabilities": [],
                "elapsed_seconds": 0.0,
                "latency": {},
                "attention": _attention_from_text(text),
                "logs": [],
                "comm_rounds": 0,
                "artifact_dir": None,
                "security_note": "",
                "error": repr(exc),
            }
            results.append(error_item)
            all_logs.append(_role_log("API", normalized_mode.upper(), repr(exc)))
    elapsed = time.perf_counter() - started
    _set_logs(all_logs)
    _set_status(
        "complete" if all(item.get("success") for item in results) else "partial",
        mode="compare",
        launch=launch,
        success_count=sum(1 for item in results if item.get("success")),
        total_count=len(results),
    )
    _record_history(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "compare",
            "launch": launch,
            "label": "",
            "confidence": "",
            "elapsed_seconds": round(elapsed, 3),
            "artifact_dir": "",
            "success_count": sum(1 for item in results if item.get("success")),
            "total_count": len(results),
        }
    )
    return {
        "success": any(item.get("success") for item in results),
        "mode": "compare",
        "launch": launch,
        "input": text,
        "results": results,
        "logs": all_logs,
        "elapsed_seconds": round(elapsed, 3),
        "success_count": sum(1 for item in results if item.get("success")),
        "total_count": len(results),
    }


def docker_status() -> dict[str, Any]:
    status = dict(_last_status)
    try:
        status["warm_service"] = _run_warm_service(["status"], timeout=60)
    except Exception as exc:
        status["warm_service"] = {"success": False, "error": repr(exc)}
    return status


def docker_logs() -> dict[str, Any]:
    return {"logs": list(_last_logs), "total_rounds": len(_last_logs), "total_bytes": sum(int(x.get("bytes", 0)) for x in _last_logs)}


def benchmark_history(limit: int = 20) -> dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {"history": []}
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    return {"history": [json.loads(line) for line in lines if line.strip()]}


def docker_service(action: str) -> dict[str, Any]:
    normalized = (action or "status").strip().lower()
    action_map = {
        "start": "start",
        "stop": "stop",
        "status": "status",
        "start_crypten": "start-crypten-service",
        "stop_crypten": "stop-crypten-service",
        "crypten_status": "crypten-service-status",
        "start_mcu": "start-mcu-service",
        "stop_mcu": "stop-mcu-service",
        "mcu_status": "mcu-service-status",
    }
    if normalized not in action_map:
        return {"success": False, "error": f"invalid action: {action}"}
    payload = _run_warm_service([action_map[normalized]], timeout=300)
    _set_status(
        "warm_service_" + normalized,
        mode="service",
        launch="docker_warm",
        warm_success=payload.get("success"),
        project=payload.get("project"),
    )
    return payload
