from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "mcu_rust" / "target" / "release" / "bert_session.exe"
EXPORT_SHARES = ROOT / "experiments" / "docker_bert_full" / "export_bert_session_shares.py"


def run(cmd: list[str], timeout: int = 300, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_modules(role: str, text: str) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"bert_module layer=(?P<layer>\d+) module=(?P<module>\S+) "
        r"elapsed_s=(?P<elapsed>[0-9.]+) units=(?P<units>\d+) units_per_s=(?P<rate>[0-9.]+)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "role": role,
                "layer": int(match.group("layer")),
                "module": match.group("module"),
                "elapsed_s": match.group("elapsed"),
                "units": int(match.group("units")),
                "units_per_s": match.group("rate"),
            }
        )
    return rows


def parse_timing(role: str, text: str) -> dict:
    match = re.search(rf"\[{re.escape(role)}\] timing (?P<body>[^\n]+)", text)
    out = {"role": role}
    if not match:
        return out
    for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", match.group("body")):
        out[key] = value
    for key, value in re.findall(r"(send_msgs|recv_msgs|send_bytes|recv_bytes|matmul_calls|cpu_matmul_calls|cuda_matmul_calls|fused_party_calls|fused_hp_calls|cuda_fallbacks)=([0-9]+)", match.group("body")):
        out[key] = int(value)
    return out


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


def communicate_roles(
    p0: subprocess.Popen,
    p1: subprocess.Popen,
    hp: subprocess.Popen,
    timeout: int,
) -> tuple[str, str, str, str, str, str]:
    procs = {"p0": p0, "p1": p1, "hp": hp}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                name: pool.submit(proc.communicate, timeout=timeout)
                for name, proc in procs.items()
            }
            done: dict[str, tuple[str, str]] = {}
            for name, future in futures.items():
                done[name] = future.result(timeout=timeout + 30)
    except Exception:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
        done = {}
        for name, proc in procs.items():
            try:
                done[name] = proc.communicate(timeout=30)
            except Exception:
                done[name] = ("", "")
        raise
    return (*done["p0"], *done["p1"], *done["hp"])


def export_real_shares(args: argparse.Namespace) -> Path:
    if args.shares_dir:
        source = Path(args.shares_dir)
        if not source.exists():
            raise FileNotFoundError(source)
        return source
    completed = run(
        [
            "python",
            str(EXPORT_SHARES),
            "--text",
            args.text,
            "--max-seq-len",
            str(args.seq),
            "--layers",
            str(args.layers),
            "--scale-bits",
            str(args.scale_bits),
        ],
        timeout=600,
    )
    match = re.search(r"\[bert-shares\] wrote (.+manifest\.json)", completed.stdout)
    if not match:
        raise RuntimeError(f"could not parse share export path:\n{completed.stdout}\n{completed.stderr}")
    return Path(match.group(1)).parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--state-mode", choices=["synthetic", "chained"], default="synthetic")
    parser.add_argument("--input-mode", choices=["synthetic", "real_io"], default="synthetic")
    parser.add_argument("--shares-dir")
    parser.add_argument("--text", default="This movie is wonderful and heartwarming.")
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--rescale-bits", type=int)
    parser.add_argument("--rescale-mode", choices=["local", "hp_clear"], default="local")
    parser.add_argument("--role-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.input_mode == "real_io" and args.state_mode != "chained":
        raise ValueError("--input-mode real_io requires --state-mode chained")
    rescale_bits = args.rescale_bits
    if rescale_bits is None:
        rescale_bits = args.scale_bits if args.input_mode == "real_io" else 0

    run(["cargo", "build", "--release", "--bin", "bert_session"], timeout=600, cwd=ROOT / "mcu_rust")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_mcu_bert_session_smoke"
    out.mkdir(parents=True, exist_ok=True)
    addr = f"127.0.0.1:{args.port}"
    shape = [
        "--batch", str(args.batch),
        "--seq", str(args.seq),
        "--hidden", str(args.hidden),
        "--heads", str(args.heads),
        "--ffn", str(args.ffn),
        "--layers", str(args.layers),
        "--state-mode", args.state_mode,
        "--input-mode", args.input_mode,
        "--scale-bits", str(args.scale_bits),
        "--rescale-bits", str(rescale_bits),
        "--rescale-mode", args.rescale_mode,
    ]
    share_root = export_real_shares(args) if args.input_mode == "real_io" else None

    hp = subprocess.Popen(
        [str(EXE), "hp", "--addr", addr, *shape],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    p0 = subprocess.Popen(
        [
            str(EXE),
            "p0",
            "--addr",
            addr,
            *shape,
            *(["--share-dir", str(share_root / "p0")] if share_root else []),
            "--out",
            str(out / "p0.out"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p1 = subprocess.Popen(
        [
            str(EXE),
            "p1",
            "--addr",
            addr,
            *shape,
            *(["--share-dir", str(share_root / "p1")] if share_root else []),
            "--out",
            str(out / "p1.out"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        p0_out, p0_err, p1_out, p1_err, hp_out, hp_err = communicate_roles(
            p0, p1, hp, args.role_timeout
        )
    except Exception as exc:
        p0_out, p0_err = p0.communicate() if p0.poll() is not None else ("", "")
        p1_out, p1_err = p1.communicate() if p1.poll() is not None else ("", "")
        hp_out, hp_err = hp.communicate() if hp.poll() is not None else ("", "")
        (out / "p0.log").write_text(p0_out + p0_err, encoding="utf-8")
        (out / "p1.log").write_text(p1_out + p1_err, encoding="utf-8")
        (out / "hp.log").write_text(hp_out + hp_err, encoding="utf-8")
        raise TimeoutError(f"role timed out or failed after {args.role_timeout}s") from exc
    (out / "p0.log").write_text(p0_out + p0_err, encoding="utf-8")
    (out / "p1.log").write_text(p1_out + p1_err, encoding="utf-8")
    (out / "hp.log").write_text(hp_out + hp_err, encoding="utf-8")
    if p0.returncode != 0 or p1.returncode != 0 or hp.returncode != 0:
        raise RuntimeError(f"role failed hp={hp.returncode} p0={p0.returncode} p1={p1.returncode}")

    module_rows = []
    module_rows.extend(parse_modules("hp", hp_out))
    module_rows.extend(parse_modules("p0", p0_out))
    module_rows.extend(parse_modules("p1", p1_out))
    timing_rows = [
        parse_timing("hp", hp_out),
        parse_timing("p0", p0_out),
        parse_timing("p1", p1_out),
    ]
    party_result = parse_party_out(out / "p0.out")
    write_csv(out / "module_timing.csv", module_rows)
    write_csv(out / "role_timing.csv", timing_rows)
    summary = [
        {
            "mode": "mcu_bert_session_smoke",
            "status": "ok",
            "batch": args.batch,
            "seq": args.seq,
            "hidden": args.hidden,
            "heads": args.heads,
            "ffn": args.ffn,
            "layers": args.layers,
            "state_mode": args.state_mode,
            "input_mode": args.input_mode,
            "scale_bits": args.scale_bits if args.input_mode == "real_io" else "",
            "rescale_bits": rescale_bits,
            "rescale_mode": args.rescale_mode,
            "critical_role_total_s": max(float(r.get("total_s", 0.0)) for r in timing_rows),
            "prediction": party_result.get("prediction", ""),
            "prediction_label": party_result.get("prediction_label", ""),
            "prob_negative": (party_result.get("probabilities") or ["", ""])[0],
            "prob_positive": (party_result.get("probabilities") or ["", ""])[1],
            "logit_negative": (party_result.get("logits") or ["", ""])[0],
            "logit_positive": (party_result.get("logits") or ["", ""])[1],
            "notes": "Local three-process TCP persistent BERT session; real_io consumes exported SST-2 embedding, mask, full encoder, pooler, and classifier shares. HP-clear bridges remain numerical baselines, not final secure protocols.",
        }
    ]
    write_csv(out / "summary.csv", summary)
    (out / "result.json").write_text(
        json.dumps(
            {
                "mode": "mcu_bert_session_smoke",
                "status": "ok",
                "result": party_result,
                "role_timing": timing_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[mcu-bert-session] wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
