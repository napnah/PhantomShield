# AegisX

基于 MCU 协议族的多方安全 Transformer 推理系统。项目当前提供三条可比较的 BERT 推理路径：

- Plaintext：标准 HuggingFace/PyTorch 明文基准。
- CrypTen：两方 Docker 通信基准，两个 rank 通过 Gloo/TCP 通信。
- MCU：三角色 Docker 通信原型，`p0 / p1 / hp` 通过 TCP 完成数值端到端 BERT 推理。

当前 MCU Docker BERT 是数值端到端原型，已经使用真实 Docker 通信，但仍包含 HP-clear rescale、fixed/real bridge、LayerNorm、tanh、reveal 等数值桥接步骤；不要把它表述为最终完整安全 BERT。

## 当前状态

- Goal 1：算子级优化完成。BERT-like batch `1,2,4` 下主要算子均达到当前 CPU Docker 接受阈值。
- Goal 2：完整 BERT Docker 数值推理完成，并已和 CrypTen Docker 对比。
- Goal 3：Dashboard 已支持选择进程启动或 Docker 启动，并可在同一输入上比较 plaintext、CrypTen、MCU。

## 目录结构

```text
AegisX/
├── dashboard/                  # FastAPI 后端和单页前端
├── docker/                     # Dockerfile 与 docker-compose.mpc.yml
├── docs/                       # 进度跟踪和技术报告
├── experiments/
│   ├── docker_real_comm/       # 算子级 Docker 对比实验
│   └── docker_bert_full/       # 完整 BERT Docker 推理脚本
├── mcu_rust/                   # Rust 协议和角色二进制
├── transformer/                # BERT、plaintext、CrypTen、MCU Python 连接层
├── scripts/                    # 模型下载和环境辅助脚本
└── results/                    # 可选的本地 benchmark 输出
```

## 部署要求

推荐优先使用 Docker 路径，减少对本机 Python/CrypTen 环境的依赖。

| 组件 | 建议 |
|---|---|
| 操作系统 | Linux、Windows 11 + Docker Desktop，或 macOS + Docker Desktop |
| Docker | Docker Engine / Docker Desktop，支持 Compose v2 |
| 宿主机 Python | 3.10+，用于编排脚本和 Dashboard 后端 |
| 宿主机 Rust | 可选；Docker 构建会在容器内编译 Rust 路径 |
| 模型文件 | `bert-base-uncased/` 和 `checkpoints/bert-sst2/` |

`bert-base-uncased/`、`checkpoints/`、`.local/` 和 Docker 运行输出体积较大，不进入 git。

## 快速部署

### 1. 克隆并进入项目

```bash
git clone https://github.com/napnah/PhantomShield.git AegisX
cd AegisX
```

GitHub 仓库名称可能仍是 `PhantomShield`；文档和界面中的项目名称为 `AegisX`。

### 2. 创建轻量宿主机 Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3. 准备模型文件

如果本机还没有模型文件，运行：

```bash
python scripts/download_models.py
```

部署到其他主机时，可以把模型放在仓库根目录：

```text
bert-base-uncased/
checkpoints/bert-sst2/
```

也可以通过环境变量指定外部模型路径：

```bash
export PHANTOM_BERT_BASE_HOST=/abs/path/bert-base-uncased
export PHANTOM_CHECKPOINTS_HOST=/abs/path/checkpoints
```

PowerShell：

```powershell
$env:PHANTOM_BERT_BASE_HOST = "D:\models\bert-base-uncased"
$env:PHANTOM_CHECKPOINTS_HOST = "D:\models\checkpoints"
```

### 4. 构建 Docker 镜像

```bash
docker compose -f docker/docker-compose.mpc.yml build crypten-r0 mcu-hp
docker build -f docker/Dockerfile.mcu.bert_session -t phantomshield-mcu:bert-session-fixed .
```

Compose 会构建 `phantomshield-crypten:local` 和默认 MCU 镜像。完整 BERT 的 MCU session 使用单独标签 `phantomshield-mcu:bert-session-fixed`。

### 5. 启动 Dashboard

```bash
python dashboard/backend/main.py
```

浏览器打开 `dashboard/frontend/index.html`。在舆情情感分析 BERT 场景中：

- 选择 BERT 模式：plaintext、CrypTen 或 MCU-Rust。
- 选择启动方式：进程启动、Docker 启动或 Docker 服务。
- 点击 `加密推理` 运行单一路径。
- 点击 `比较三路` 在同一输入上比较 plaintext、CrypTen 和 MCU。

## 命令行启动方式

### 进程模式

进程模式适合快速本地检查，但不能证明真实容器间通信。

```bash
python - <<'PY'
import sys
sys.path[:0] = ["dashboard/backend", "."]
from bert_orchestrator import compare
out = compare("This movie is wonderful and heartwarming.", launch="process", max_seq_len=16)
print(out["success"], out["success_count"], out["total_count"])
for item in out["results"]:
    print(item["mode"], item["label"], item["elapsed_seconds"])
PY
```

### 通过 Dashboard 后端运行 Docker 三路比较

```bash
python - <<'PY'
import sys
sys.path[:0] = ["dashboard/backend", "."]
from bert_orchestrator import compare
out = compare("This movie is wonderful and heartwarming.", launch="docker", max_seq_len=16)
print(out["success"], out["success_count"], out["total_count"])
for item in out["results"]:
    print(item["mode"], item["label"], item["elapsed_seconds"], item.get("artifact_dir"))
PY
```

### 常驻 Docker 服务模式

常驻模式会先把 CrypTen 与 MCU 的 Docker 容器拉起并保持运行。当前有两级 warm/service 路径：

- `docker_warm`：容器常驻，主进程通过 `docker compose exec` 触发推理；rank/role 进程仍是每次请求启动。
- `docker_service`：CrypTen 已实现 v2，两个 rank 进程和 BERT 模型/tokenizer 常驻，主进程通过 request/response JSON 触发多次推理；MCU 已实现 v2.1，`hp/p0/p1` role wrapper 进程常驻，静态模型 share 会缓存，输入 share 由 host 侧常驻 exporter 生成，但每次请求内部仍会重新建立协议连接并读取 request/model share，尚不是最终的 in-process persistent session。

```bash
python experiments/docker_bert_full/warm_docker_service.py start
python experiments/docker_bert_full/warm_docker_service.py start-crypten-service
python experiments/docker_bert_full/warm_docker_service.py start-mcu-service
python experiments/docker_bert_full/warm_docker_service.py status
python experiments/docker_bert_full/warm_docker_service.py crypten-service-status
python experiments/docker_bert_full/warm_docker_service.py mcu-service-status
python experiments/docker_bert_full/warm_docker_service.py infer-crypten --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-crypten-service --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-mcu --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-mcu-service --text "This movie is wonderful and heartwarming."
```

也可以从后端统一接口调用：

```bash
python - <<'PY'
import sys
sys.path[:0] = ["dashboard/backend", "."]
from bert_orchestrator import compare, docker_service
print(docker_service("start")["success"])
print(docker_service("start_crypten")["success"])
print(docker_service("start_mcu")["success"])
out = compare("This movie is wonderful and heartwarming.", launch="docker_service", max_seq_len=16, modes=["crypten", "mcu_rust"])
print(out["success"], out["success_count"], out["total_count"])
for item in out["results"]:
    print(item["mode"], item["label"], item["elapsed_seconds"], item["latency"])
PY
```

### CrypTen Docker BERT

```bash
python experiments/docker_bert_full/run_docker_bert_comparison.py \
  --samples 1 \
  --repeat 1 \
  --max-seq-len 16 \
  --max-layers 12 \
  --crypten-nonlinear native \
  --skip-build
```

### MCU Docker BERT

```bash
python experiments/docker_bert_full/run_mcu_bert_session_docker.py \
  --image phantomshield-mcu:bert-session-fixed \
  --batch 1 \
  --seq 16 \
  --hidden 768 \
  --heads 12 \
  --ffn 3072 \
  --layers 12 \
  --state-mode chained \
  --input-mode real_io \
  --text "This movie is wonderful and heartwarming." \
  --scale-bits 16 \
  --rescale-mode hp_clear
```

### 算子级 Docker benchmark

```bash
python experiments/docker_real_comm/run_docker_comparison.py \
  --preset bert \
  --repeat 3 \
  --batches 1,2,4 \
  --skip-build
```

`--skip-build` 只适合镜像已经构建好的机器；新主机部署时请去掉该参数。`bert` preset 对应下方表格使用的 BERT-base-like 算子形状。

## 可迁移部署建议

- 不要在脚本或文档中写死本机盘符。模型和输出目录优先使用环境变量指定。
- Windows 上 Docker bind mount 有时更适合使用较短路径；当前 runner 会在需要时自动创建临时 Docker 挂载目录。
- 如果希望所有生成文件都留在项目内，可以设置：

```bash
export PHANTOM_DOCKER_BIND_HOST="$PWD/.local/docker_out"
export PHANTOM_DOCKER_BERT_SHARES_HOST="$PWD/.local/bert_shares"
export PHANTOM_DOCKER_MODEL_CACHE="$PWD/.local/docker_models"
```

PowerShell：

```powershell
$env:PHANTOM_DOCKER_BIND_HOST = "$PWD\.local\docker_out"
$env:PHANTOM_DOCKER_BERT_SHARES_HOST = "$PWD\.local\bert_shares"
$env:PHANTOM_DOCKER_MODEL_CACHE = "$PWD\.local\docker_models"
```

## 最新效率快照

以下比值均为 `MCU 耗时 / CrypTen 耗时`；小于 `1.0x` 表示该测量中 MCU 更快。

### 算子级 Docker 对比

数据来源：`experiments/20260628_211821_docker_real_comm/operator_ratio_matrix.csv`

实验范围：真实 Docker 通信；MCU 为 `p0/p1/hp` TCP；CrypTen 为两 rank Gloo/TCP；形状为 BERT-base-like；后端为 CPU Docker 路径。

| 算子 | Batch 1 | Batch 2 | Batch 4 | 状态 |
|---|---:|---:|---:|---|
| `elemul` | `1.00x` | `0.89x` | `0.98x` | accepted |
| `matmul` | `1.27x` | `1.12x` | `1.02x` | accepted |
| `exp` | `1.04x` | `1.48x` | `1.51x` | accepted |
| `sigmoid` | `0.31x` | `0.43x` | `0.44x` | accepted |
| `gelu` | `0.38x` | `0.35x` | `0.50x` | accepted |
| `softmax` | `0.16x` | `0.19x` | `0.23x` | accepted |

### BERT 推理 Docker 对比

数据来源：

- 10 样本 benchmark：`experiments/20260629_180900_goal2_docker_bert_comparison/summary.csv`
- Goal3 单输入 Docker smoke：
  - CrypTen：`experiments/20260630_101258_docker_bert_full/summary.csv`
  - MCU：`experiments/20260630_101402_mcu_bert_session_docker/summary.csv`

| 路径 | 启动形式 | 样本数 | 平均延迟 / 关键角色延迟 | 准确率或预测 | 说明 |
|---|---|---:|---:|---|---|
| Plaintext | 宿主机进程 | 10 | `0.049s/sample` | `1.00` accuracy | 明文基准，不提供 MPC 隐私保护。 |
| CrypTen | Docker，两 rank | 10 | `11.656s/sample` | `0.90` accuracy | Gloo/TCP 上的 native nonlinear BERT 路径。 |
| MCU | Docker，`p0/p1/hp` | 10 | `6.680s/sample` | `1.00` accuracy | HP-clear 数值 BERT 原型。 |
| Plaintext | Dashboard Docker 对比基准 | 1 | `0.179s` | positive | Docker 对比 UI 中使用的宿主机明文基准。 |
| CrypTen | Dashboard Docker 对比 | 1 | `18.011s` | positive | 真实 Docker 双 rank 运行。 |
| MCU | Dashboard Docker 对比 | 1 | `23.773s` | positive | 真实 Docker 三角色运行。 |

解读：

- 10 样本结果更适合作为摊销后的端到端 benchmark。
- 单输入 Dashboard 结果包含冷启动和编排开销，更适合作为部署 smoke test。
- 当前 MCU 速度结果对应数值原型，不代表最终完整安全 BERT。

## 安全边界

已经实现并测量：

- MCU `p0/p1/hp` 真实 TCP 通信。
- CrypTen 两 rank 真实 Docker 通信。
- 张量级 matmul 和批量非线性算子。
- 使用真实 checkpoint 权重和真实输入 embedding 的完整 BERT 数值 Docker 流程。

尚未达到最终安全版本：

- MCU 完整 BERT 仍使用 HP-clear 数值桥接，包括 rescale/fixed-real conversion、非线性反馈、LayerNorm 和最终分类输出。
- 后续安全目标应替换为 wrap-correct secure truncation/rescale、安全 fixed/real conversion、安全 LayerNorm，并定义清晰的 reveal policy。

## 常见问题

Docker daemon 不可用：

```bash
docker info
```

运行 Docker 路径前，请先启动 Docker Desktop 或 Docker Engine。

缺少模型文件：

```bash
python scripts/download_models.py
```

也可以设置 `PHANTOM_BERT_BASE_HOST` 和 `PHANTOM_CHECKPOINTS_HOST`。

清理残留容器：

```bash
docker compose -f docker/docker-compose.mpc.yml down --remove-orphans
```

Dashboard 导入路径问题：

```bash
PYTHONPATH=dashboard/backend:. python dashboard/backend/main.py
```

PowerShell：

```powershell
$env:PYTHONPATH = "dashboard/backend;."
python dashboard/backend/main.py
```
