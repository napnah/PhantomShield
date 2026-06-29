from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.mpc.yml"
EXPORT_SHARES = ROOT / "experiments" / "docker_bert_full" / "export_bert_session_shares.py"


def compose(project: str) -> list[str]:
    return ["docker", "compose", "-p", project, "-f", str(COMPOSE)]


def run(cmd: list[str], env: dict[str, str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout, check=check)


def cleanup(project: str, env: dict[str, str]) -> None:
    run([*compose(project), "down", "--remove-orphans"], env, timeout=120, check=False)


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


def docker_bind_root(out: Path, timestamp: str) -> Path:
    override = os.environ.get("PHANTOM_DOCKER_BIND_HOST")
    if override:
        path = Path(override)
    elif os.name == "nt":
        path = Path(tempfile.gettempdir()) / "phantomshield_mcu_bert_session" / timestamp
    else:
        path = out / "shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def docker_share_root(out: Path, timestamp: str) -> Path:
    override = os.environ.get("PHANTOM_DOCKER_BERT_SHARES_HOST")
    if override:
        path = Path(override)
    elif os.name == "nt":
        path = Path(tempfile.gettempdir()) / "phantomshield_mcu_bert_shares" / timestamp
    else:
        path = out / "bert_shares_bind"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_real_shares(args: argparse.Namespace, out: Path, timestamp: str) -> Path:
    if args.shares_dir:
        source = Path(args.shares_dir)
        if not source.exists():
            raise FileNotFoundError(source)
    else:
        cmd = [
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
        ]
        completed = run(cmd, os.environ.copy(), timeout=600)
        match = re.search(r"\[bert-shares\] wrote (.+manifest\.json)", completed.stdout)
        if not match:
            raise RuntimeError(f"could not parse share export path:\n{completed.stdout}\n{completed.stderr}")
        source = Path(match.group(1)).parent

    local_copy = out / "bert_shares"
    if local_copy.exists():
        shutil.rmtree(local_copy)
    shutil.copytree(source, local_copy)

    bind = docker_share_root(out, timestamp)
    if bind.exists():
        shutil.rmtree(bind, ignore_errors=True)
    shutil.copytree(source, bind)
    return bind


def parse_modules(role: str, text: str) -> list[dict]:
    pattern = re.compile(
        r"bert_module layer=(?P<layer>\d+) module=(?P<module>\S+) "
        r"elapsed_s=(?P<elapsed>[0-9.]+) units=(?P<units>\d+) units_per_s=(?P<rate>[0-9.]+)"
    )
    return [
        {
            "role": role,
            "layer": int(m.group("layer")),
            "module": m.group("module"),
            "elapsed_s": m.group("elapsed"),
            "units": int(m.group("units")),
            "units_per_s": m.group("rate"),
        }
        for m in pattern.finditer(text)
    ]


def parse_timing(role: str, text: str) -> dict:
    match = re.search(rf"\[{re.escape(role)}\] timing (?P<body>[^\n]+)", text)
    row = {"role": role}
    if not match:
        return row
    body = match.group("body")
    for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", body):
        row[key] = value
    for key, value in re.findall(
        r"(send_msgs|recv_msgs|send_bytes|recv_bytes|matmul_calls|cpu_matmul_calls|cuda_matmul_calls|fused_party_calls|fused_hp_calls|cuda_fallbacks)=([0-9]+)",
        body,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--image", default="phantomshield-mcu:bert-session")
    parser.add_argument("--project", default="psbertmcu")
    parser.add_argument("--state-mode", choices=["synthetic", "chained"], default="synthetic")
    parser.add_argument("--input-mode", choices=["synthetic", "real_io"], default="synthetic")
    parser.add_argument("--shares-dir")
    parser.add_argument("--text", default="This movie is wonderful and heartwarming.")
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--rescale-bits", type=int)
    parser.add_argument("--rescale-mode", choices=["local", "hp_clear"], default="local")
    args = parser.parse_args()
    if args.input_mode == "real_io" and args.state_mode != "chained":
        raise ValueError("--input-mode real_io requires --state-mode chained")
    rescale_bits = args.rescale_bits
    if rescale_bits is None:
        rescale_bits = args.scale_bits if args.input_mode == "real_io" else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_mcu_bert_session_docker"
    out.mkdir(parents=True, exist_ok=True)
    bind = docker_bind_root(out, timestamp)
    share_bind = None
    if args.input_mode == "real_io":
        share_bind = export_real_shares(args, out, timestamp)

    env = os.environ.copy()
    env.update(
        {
            "PHANTOM_OUT_HOST": str(bind),
            "MCU_IMAGE": args.image,
            "MCU_KIND": "bert_session",
            "MCU_PORT": "9400",
            "MCU_BATCH": str(args.batch),
            "MCU_SEQ": str(args.seq),
            "MCU_HIDDEN": str(args.hidden),
            "MCU_HEADS": str(args.heads),
            "MCU_FFN": str(args.ffn),
            "MCU_LAYERS": str(args.layers),
            "MCU_STATE_MODE": args.state_mode,
            "MCU_INPUT_MODE": args.input_mode,
            "MCU_SCALE_BITS": str(args.scale_bits),
            "MCU_RESCALE_BITS": str(rescale_bits),
            "MCU_RESCALE_MODE": args.rescale_mode,
            "MCU_OUT_DIR": "/workspace/out/run",
            "MCU_START_DELAY": "1",
        }
    )
    if share_bind is not None:
        env["PHANTOM_BERT_SHARES_HOST"] = str(share_bind)

    cleanup(args.project, env)
    p0 = p1 = None
    try:
        run([*compose(args.project), "up", "-d", "mcu-hp"], env, timeout=120)
        time.sleep(1.0)
        p0 = subprocess.Popen(
            [*compose(args.project), "run", "--rm", "-T", "mcu-p0"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p1 = subprocess.Popen(
            [*compose(args.project), "run", "--rm", "-T", "mcu-p1"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p0_out, p0_err = p0.communicate(timeout=300)
        p1_out, p1_err = p1.communicate(timeout=300)
        hp_logs = run([*compose(args.project), "logs", "--no-color", "mcu-hp"], env, timeout=120, check=False).stdout
        verify = run([*compose(args.project), "run", "--rm", "-T", "mcu-verify"], env, timeout=120, check=False)
    finally:
        if p0 and p0.poll() is None:
            p0.kill()
        if p1 and p1.poll() is None:
            p1.kill()
        cleanup(args.project, env)

    (out / "p0.log").write_text(p0_out + p0_err, encoding="utf-8")
    (out / "p1.log").write_text(p1_out + p1_err, encoding="utf-8")
    (out / "hp.log").write_text(hp_logs, encoding="utf-8")
    (out / "verify.log").write_text(verify.stdout + verify.stderr, encoding="utf-8")

    if p0.returncode != 0 or p1.returncode != 0:
        raise RuntimeError(f"p0/p1 failed: p0={p0.returncode}, p1={p1.returncode}")
    if verify.returncode != 0:
        raise RuntimeError(f"verify failed: {verify.stdout}\n{verify.stderr}")

    if bind != out / "shared" and bind.exists():
        shutil.copytree(bind, out / "shared", dirs_exist_ok=True)
        shutil.rmtree(bind, ignore_errors=True)
    if share_bind is not None and share_bind.exists() and share_bind != out / "bert_shares_bind":
        shutil.rmtree(share_bind, ignore_errors=True)

    module_rows = []
    module_rows.extend(parse_modules("hp", hp_logs))
    module_rows.extend(parse_modules("p0", p0_out))
    module_rows.extend(parse_modules("p1", p1_out))
    timing_rows = [parse_timing("hp", hp_logs), parse_timing("p0", p0_out), parse_timing("p1", p1_out)]
    write_csv(out / "module_timing.csv", module_rows)
    write_csv(out / "role_timing.csv", timing_rows)
    critical = max(float(r.get("total_s", 0.0)) for r in timing_rows)
    party_result = parse_party_out(out / "shared" / "run" / "p0.out")
    write_csv(
        out / "summary.csv",
        [
            {
                "mode": "mcu_bert_session_docker",
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
                "critical_role_total_s": f"{critical:.9f}",
                "prediction": party_result.get("prediction", ""),
                "prediction_label": party_result.get("prediction_label", ""),
                "prob_negative": (party_result.get("probabilities") or ["", ""])[0],
                "prob_positive": (party_result.get("probabilities") or ["", ""])[1],
                "logit_negative": (party_result.get("logits") or ["", ""])[0],
                "logit_positive": (party_result.get("logits") or ["", ""])[1],
                "image": args.image,
                "notes": "Docker three-role TCP persistent BERT session; real_io consumes exported SST-2 embedding, mask, full encoder, pooler, and classifier shares. HP-clear bridges remain numerical baselines, not final secure protocols.",
            }
        ],
    )
    (out / "result.json").write_text(
        json.dumps(
            {
                "mode": "mcu_bert_session_docker",
                "status": "ok",
                "image": args.image,
                "result": party_result,
                "role_timing": timing_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[mcu-bert-docker] wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
