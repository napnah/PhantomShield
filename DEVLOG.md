# PhantomShield 开发日志（DEVLOG）

> 本文档记录 PhantomShield（MCU-Transformer）系统的开发历程，按时间顺序整理各阶段完成的工作。
> 作者负责模块：底层通信框架、密文推理集成、实验验证、Web Dashboard、系统统筹。

---

## 阶段一：环境搭建与技术选型（6月4日）

**目标**：搭建可运行 MPC 推理的开发环境。

完成内容：
- 确定技术栈：Python 3.11.9 + PyTorch 2.5.1（CUDA 12.1）+ CrypTen + transformers
- 创建 venv 虚拟环境，配置 GPU 加速（RTX 4060 Laptop）
- 解决 PyTorch GPU 版本国内下载问题（手动下载 cu121 wheel 离线安装）
- 修复 CrypTen 0.4.1 与 PyTorch 2.5 的兼容性问题（手动 patch onnx_converter.py）
- 修复 CrypTen 单进程模式下 Beaver 乘法的 all_reduce 接口 bug
- 通过 ModelScope 下载 BERT-Base / BERT-Large（HuggingFace 国内访问受限）

**踩坑记录**：
- CrypTen 从源码安装会自动拉取 CPU 版 torch，覆盖 GPU 版，需重装
- 版本兼容矩阵：torch 2.5.1+cu121 / transformers 4.46 / pyarrow 17 / datasets 2.21

---

## 阶段二：MCU 核心协议实现（6月6日）

**目标**：实现 MCU 范式的乘法协议，验证密码学正确性。

完成内容：
- `mcu_core/prg_sync.py`：基于 AES-CTR 的同步伪随机数生成器（PRG）
- `mcu_core/comm.py`：三方 socket 通信层（4字节长度前缀 + JSON）
- `mcu_core/protocols/multiply.py`：MCU 安全乘法协议 Π_mul（2轮通信）
  - 推导并验证核心恒等式：x·y = (x+r_x)(y+r_y) - x·r_y - y·r_x - r_x·r_y
  - 5 组随机测试全部通过（如 12345×67890=838102050）
- `mcu_core/mock_comm.py`：本地测试用 mock 通信接口（基于线程安全 Queue），供队友免联调测试协议

**关键技术点**：理解 MCU 的 Mask-Compute-Unmask 范式——P0 用公开掩码保护份额，P1 直接发份额（因 x0 随机故安全），HP 在掩码数据上计算。

---

## 阶段三：三进程通信框架（6月6日）

**目标**：从单进程模拟升级为真实的多进程 MPC。

完成内容：
- `party.py`：三进程启动入口（`--role hp/p0/p1`）
- 实现 HP/P0/P1 三个独立进程，通过 localhost socket 通信
- 端到端验证：三个独立进程协同完成 MPC 乘法，结果正确（12345×67890=838102050）

**意义**：这是"真 MPC"与"单进程模拟"的本质区别——三个进程内存隔离，只能通过 socket 通信，任何单方无法访问对方数据。

---

## 阶段四：密文线性层（6月6日）

**目标**：实现 Transformer 最基础的密文矩阵乘法。

完成内容：
- `transformer/mcu_linear.py`：密文线性层 Y=XW
  - 定点数方案：SCALE=2^16，矩阵乘法结果除以 SCALE²
  - 正确处理有符号整数与负数还原
- `experiments/test_bert_linear.py`：真实 BERT-Base 的 W_Q 权重密文矩阵乘法
  - 256 次 MPC 乘法，最大误差 2.16e-5，验证通过

**踩坑记录**：PRG 生成的随机数超出 PyTorch long 范围会溢出，需限制在有符号 64 位内。

---

## 阶段五：性能基准与演示优化（6月6日）

完成内容：
- `experiments/benchmark_linear.py`：不同矩阵规模的密文线性层耗时测量
  - 发现纯 Python 逐元素实现的性能瓶颈（64×64 需 1.7 秒）
- `experiments/tiny_bert_demo.py`：微型 BERT（2层/32维/4头）快速演示
  - 密文 W_Q 投影 0.91 秒，适合答辩现场演示
  - 直观展示 P1 看到的随机份额 vs 真实值

**决策**：演示采用"微型模型快速跑通 + 完整模型性能数据"双轨策略。

---

## 阶段六：完整密文 BERT 推理（6月12日）

**目标**：把 Attention、FFN、LayerNorm 串成完整的密文推理。

完成内容：
- `transformer/mcu_attention_crypten.py`：密文 Self-Attention（含 Softmax）
- `transformer/mcu_ffn_crypten.py`：密文 FFN（GeLU 用 sigmoid 近似实现）
- `transformer/mcu_bert_crypten.py`：完整密文 Encoder Layer
  - 串联 Attention + FFN + LayerNorm + 残差连接
  - 2 层端到端推理，误差 1.8e-3，验证通过

**意义**：系统现在完整可运行。代码中标记 ★ 的位置（Softmax/GeLU）即为 MCU 协议的替换接口，潘涵的协议完成后直接替换。

**踩坑记录**：
- CrypTen 无内置 gelu()，用 x·sigmoid(1.702x) 近似
- CrypTen 的 pow() 不支持负小数指数，LayerNorm 的 1/√var 改用揭示方差（统计量可安全揭示）后明文计算

---

## 阶段七：GLUE 基线实验（6月12日）

**目标**：复现 SecFormer 的 GLUE 基线，作为对比数据。

完成内容：
- `experiments/run_glue_baseline.py`：BERT-Base/Large 的 GLUE 微调评测
- 跑出 6 个任务的基线数据：
  - BERT-Base：CoLA 53.9 / RTE 66.8 / MRPC 87.4 / STS-B 88.9 / QNLI 91.4
  - BERT-Large CoLA：60.7
- 数据与 SecFormer 论文参考值高度吻合，验证训练流程正确

---

## 阶段八：核心对比实验——2Quad 精度归零（6月14日）

**目标**：证明 SecFormer 的 2Quad 近似导致精度归零，凸显 MCU 精确 Softmax 的价值。

完成内容：
- `experiments/verify_2quad_collapse.py`：微调 CoLA → 原始 Softmax 评测 → 2Quad 替换后评测
- **核心结果**：BERT-Large CoLA 原始 Softmax **59.92** → 2Quad 近似 **0.0**
- 这是自主复现的实验数据（非引用论文），是作品最有力的论据

---

## 阶段九：CrypTen 延迟基线（6月14日）

完成内容：
- `experiments/measure_crypten_latency.py`：密文线性层延迟测量
- 数据：Q投影 566ms，FFN 第一层 3.2 秒，FFN 第二层 4.5 秒
- 减速比 194x ~ 2235x，直观说明密文推理的计算开销来源

---

## 阶段十：Web Dashboard（6月14日）

**目标**：可视化演示系统，用于答辩展示。

完成内容：
- `dashboard/backend/main.py`：FastAPI 后端，提供推理/日志/性能/隐私/对比接口
- `dashboard/backend/infer_engine.py`：真实密文推理引擎 + 场景规则分类
  - 真实跑 256 次 MPC 乘法产生秘密份额
  - 基于关键词规则的合理且可复现的分类
- `dashboard/frontend/index.html`：单文件前端
  - 三场景切换（医疗诊断 / 金融欺诈 / 舆情分析）
  - 富输出：概率分布、注意力高亮
  - 隐私泄露对比（明文推理 vs MCU 推理，服务商所见对比）
  - 三方通信日志实时动画
  - 后端离线时自动降级为模拟数据，保证演示稳定

---

## 阶段十一：集成 MCU 协议套件（6月14日）

**目标**：把队友潘涵实现的 MCU 非线性协议接入系统，完成从"CrypTen 内置函数"到"自研 MCU 协议"的升级。

完成内容：
- 接收并集成潘涵实现的协议：`wrap_detect.py`、`exponential.py`、`softmax.py`、`gelu.py`
- 在本机环境跑通协议全量测试 `run_all_tests.py`，六协议全部通过：
  - Π_mul 精确，Wrap 检测 8/8，Π_exp 误差 ~1.7e-11
  - Π_softmax 误差 ~1.3e-14，Π_sigmoid 误差 ~7.9e-15，Π_gelu 误差 ~2.1e-12
- 编写 `experiments/verify_mcu_protocols_integration.py`：用真实 MCU 协议（非 CrypTen 内置）跑通 Attention 的 Softmax 和 FFN 的 GeLU
  - 真 Π_softmax 计算 Attention：最大误差 3.66e-15，概率和精确为 1
  - 真 Π_gelu 计算 FFN：最大误差 3.69e-12

**意义**：至此 PhantomShield 的非线性计算全部由自研 MCU 协议精确支撑，作品从"基于 CrypTen 搭建"升级为"基于 MCU 范式的原创实现"，创新性闭环完成。系统采用双轨设计：CrypTen 版用于流畅的现场演示，MCU 协议版用于证明协议真实性与精度。

---

## 团队协作记录

为队友潘涵打包了协议开发工具包（mcu_core 核心模块 + mock_comm + 使用说明），
使其能在不依赖完整通信框架的情况下，本地独立开发和测试 MCU 协议。
潘涵交付四个核心协议（wrap/exp/softmax/gelu）后，由本人完成集成与端到端验证。

接手并完成了原属队友的对比实验部分（SecFormer GLUE 复现、2Quad 归零验证、CrypTen 延迟基线）。

为队友郑意准备了项目说明文档与 Web Dashboard，供其了解项目、操作演示与准备答辩。

---

## 累计交付清单

**代码模块**：
- MCU 协议层：prg_sync / comm / mock_comm / multiply
- 通信框架：party.py（三进程）
- 密文推理层：mcu_linear / mcu_attention / mcu_ffn / mcu_bert
- 实验脚本：6 个（基线/演示/基准/归零验证/延迟测量）
- Web Dashboard：后端 + 推理引擎 + 前端

**实验数据**：
- GLUE 基线（6 任务）
- 2Quad 精度归零（59.92 → 0.0）
- CrypTen 密文延迟基线
- 各协议正确性验证（误差 < 2e-5）

**文档**：
- 详细技术方案、前置知识手册、完整技术方案、作品报告
- 队友任务说明（潘涵协议包、余果果实验指南）
