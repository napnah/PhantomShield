# PhantomShield

**基于 MCU 架构与多方安全计算的隐私保护 Transformer 推理系统**

PhantomShield 让"持有数据的一方"与"持有模型的一方"在双方私有数据均不泄露的前提下，
协同完成 Transformer（BERT）推理。系统基于 MCU（Mask-Compute-Unmask）范式实现，
自研了完整的非线性函数安全计算协议套件，无需任何离线预处理。

---

## 一、项目亮点

- **真实三方架构**：P0（用户方）、P1（服务商）、HP（辅助方）三个独立进程通过 TCP socket 通信，物理隔离，非单进程模拟。
- **自研 MCU 协议套件**：安全乘法、指数、Softmax、Sigmoid、GeLU 协议全部自主实现，精度达浮点舍入极限（Softmax 误差 ~1e-14）。
- **精确 Softmax**：不采用 SecFormer 的 2Quad 近似，保留精确计算。实验证明 2Quad 近似会导致 BERT-Large 在 CoLA 任务上精度归零（59.92 → 0）。
- **零预处理**：彻底消除 Beaver 三元组预处理开销，随机性全部由同步 PRG 在线生成。
- **可视化演示**：Web Dashboard 支持医疗/金融/舆情三场景，实时展示通信日志与隐私泄露对比。

---

## 二、环境要求

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.1（GPU）|
| transformers | 4.46.0 |
| datasets | 2.21.0 |
| pyarrow | 17.0.0 |
| CrypTen | 0.4.1（需手动修复，见下）|
| 其他 | pycryptodome, fastapi, uvicorn, scikit-learn, scipy, modelscope |

---

## 三、安装步骤

### 1. 创建虚拟环境

```powershell
cd PhantomShield
python -m venv venv
venv\Scripts\activate
```

### 2. 安装 PyTorch（GPU 版）

```powershell
pip install torch-2.5.1+cu121-cp311-cp311-win_amd64.whl
```

无 GPU 可改用：`pip install torch==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 安装其余依赖

```powershell
pip install transformers==4.46.0 datasets==2.21.0 pyarrow==17.0.0 ^
            pycryptodome fastapi uvicorn scikit-learn scipy modelscope ^
            -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 修复 CrypTen 兼容性

CrypTen 0.4.1 与 PyTorch 2.5 存在兼容问题，需手动修改：

打开 `venv\Lib\site-packages\crypten\nn\onnx_converter.py`，约第 30 行，
将 except 块中的 `from torch.onnx._internal.registration import registry` 一行删除，
仅保留 `SYM_REGISTRY = False`。

### 5. 下载 BERT 模型

```powershell
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bert-base-uncased', local_dir='./bert-base-uncased')"
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bert-large-uncased', local_dir='./bert-large-uncased')"
```

---

## 四、目录结构

```
PhantomShield/
├── mcu_core/                    # MCU 核心协议层
│   ├── prg_sync.py              # 同步伪随机数生成器（AES-CTR）
│   ├── comm.py                  # 三方 socket 通信
│   ├── mock_comm.py             # 本地测试用 mock 通信
│   └── protocols/               # MCU 协议套件
│       ├── multiply.py          # Π_mul 安全乘法
│       ├── wrap_detect.py       # Wrap 溢出检测
│       ├── exponential.py       # Π_exp 安全指数
│       ├── softmax.py           # Π_softmax 安全 Softmax
│       ├── gelu.py              # Π_sigmoid / Π_gelu
│       └── run_all_tests.py     # 协议全量测试
├── transformer/                 # 密文推理层
│   ├── mcu_linear.py            # 密文线性层
│   ├── mcu_attention_crypten.py # 密文 Attention
│   ├── mcu_ffn_crypten.py       # 密文 FFN
│   └── mcu_bert_crypten.py      # 完整密文 BERT
├── experiments/                 # 实验脚本
│   ├── run_glue_baseline.py     # GLUE 基线
│   ├── verify_2quad_collapse.py # 2Quad 精度归零验证
│   ├── measure_crypten_latency.py        # 延迟基线
│   └── verify_mcu_protocols_integration.py # MCU 协议集成验证
├── dashboard/                   # Web 可视化
│   ├── backend/main.py          # FastAPI 后端
│   ├── backend/infer_engine.py  # 推理引擎
│   └── frontend/index.html      # 前端页面
├── party.py                     # 三进程启动入口
└── results/                     # 实验结果
```

---

## 五、运行说明

### 1. 验证 MCU 协议套件（核心）

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m mcu_core.protocols.run_all_tests
```

预期：6 个协议全部通过，Softmax 误差 ~1e-14，GeLU 误差 ~1e-12。

### 2. 验证 MCU 协议支撑 Transformer 非线性层

```powershell
python experiments\verify_mcu_protocols_integration.py
```

预期：真 Π_softmax / Π_gelu 计算的 Attention 与 FFN 结果正确。

### 3. 三进程 MPC 演示

开三个终端，分别运行：

```powershell
python party.py --role hp
python party.py --role p0
python party.py --role p1
```

### 4. 完整密文 BERT 推理

```powershell
python transformer\mcu_bert_crypten.py
```

### 5. 启动 Web Dashboard

```powershell
cd dashboard\backend
python main.py
```

然后用浏览器打开 `dashboard\frontend\index.html`，访问推理演示界面。
（后端未启动时前端会自动降级为模拟数据，演示仍可进行。）

---

## 六、复现关键实验

```powershell
# GLUE 基线（BERT-Base 五任务 + BERT-Large CoLA）
python experiments\run_glue_baseline.py

# 核心实验：2Quad 近似导致精度归零（59.92 → 0）
python experiments\verify_2quad_collapse.py

# CrypTen 密文推理延迟基线
python experiments\measure_crypten_latency.py
```

实验结果保存在 `results/` 目录。

---

## 七、常见问题

**Q：出现 conda 的 (base) 环境干扰？**
先 `conda deactivate` 再 `venv\Scripts\activate`，确保只有 (venv)。

**Q：控制台中文或符号乱码？**
设置 `$env:PYTHONIOENCODING="utf-8"`。

**Q：torch 提示 CPU 版 / CUDA 不可用？**
确认安装的是 cu121 版本：`python -c "import torch; print(torch.cuda.is_available())"` 应为 True。

**Q：transformers 报 float8 相关错误？**
版本过新，降级到 4.46.0。

---

## 八、安全模型

系统遵循半诚实（Semi-honest）+ HP 非合谋安全模型：P0、P1 严格执行协议但会尝试推断对方数据；
HP 不与任何一方合谋。HP 在计算时仅见加性/乘性掩码后的数据，对真实数据无法推断。
代数函数（乘法）在整数环 Z_{2^64} 上精确计算；超越函数（exp/softmax/gelu）在浮点域精确计算，
仅含浮点舍入误差，无多项式近似误差。
