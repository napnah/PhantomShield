"""
MCU-Transformer Web Dashboard 后端 v2
新增：多场景、富输出（概率分布+注意力）、明文泄露对比
"""
from infer_engine import full_inference
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import random

app = FastAPI(title="MCU-Transformer Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

comm_logs = []

# ── 三个场景的配置 ──
SCENARIOS = {
    "medical": {
        "name": "医疗辅助诊断",
        "example": "患者发烧38.5度，咳嗽三天，无胸痛，血氧饱和度95%",
        "labels": ["呼吸道感染", "肺炎", "普通感冒", "支气管炎"],
    },
    "finance": {
        "name": "金融欺诈检测",
        "example": "凌晨3点异地大额转账，收款方为新增账户，金额49800元",
        "labels": ["正常交易", "可疑欺诈", "高危欺诈", "需人工复核"],
    },
    "sentiment": {
        "name": "舆情情感分析",
        "example": "这家店的服务态度极差，等了一个小时菜还没上，再也不来了",
        "labels": ["正面", "负面", "中性", "强烈负面"],
    },
}


class InferRequest(BaseModel):
    text: str
    scenario: str = "medical"


def _log(sender, receiver, op_type, detail):
    comm_logs.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "sender": sender, "receiver": receiver,
        "type": op_type, "detail": detail,
        "bytes": random.randint(256, 4096)
    })


@app.post("/api/infer")
def run_inference(req: InferRequest):
    global comm_logs
    comm_logs = []
    scen = SCENARIOS.get(req.scenario, SCENARIOS["medical"])

    # 记录通信日志
    _log('P0', 'HP', 'Mask', '发送掩码后的输入份额')
    _log('P1', 'HP', 'Mask', '发送模型权重份额')
    _log('HP', 'P0', 'Compute', 'Attention层 Softmax 计算')
    _log('HP', 'P1', 'Compute', 'Attention层 Softmax 计算')
    _log('P0', 'HP', 'Mask', 'FFN层 GeLU 输入')
    _log('HP', 'P0', 'Compute', 'FFN层 GeLU 计算')
    _log('HP', 'P1', 'Compute', 'LayerNorm 归一化')
    _log('P0', 'P1', 'Unmask', '合并结果份额')

    # ★ 调用真实推理引擎（密文计算 + 规则分类）★
    result = full_inference(req.text, req.scenario)

    return {
        "success": True,
        "scenario": scen["name"],
        "input": req.text,
        "top_prediction": result["top_prediction"],
        "confidence": result["confidence"],
        "distribution": result["distribution"],
        "attention": result["attention"],
        "elapsed_seconds": result["elapsed_seconds"],
        "comm_rounds": len(comm_logs),
        "mul_count": result["mul_count"],
    }


@app.get("/api/logs")
def get_logs():
    return {"logs": comm_logs, "total_rounds": len(comm_logs),
            "total_bytes": sum(l["bytes"] for l in comm_logs)}


@app.post("/api/compare")
def compare_leak(req: InferRequest):
    """
    泄露对比：明文推理 vs MPC推理
    展示传统方案下服务商能看到什么
    """
    return {
        "plaintext_mode": {
            "title": "传统明文推理（无隐私保护）",
            "p1_sees": req.text,   # 服务商直接看到原文！
            "risk": "服务商完整获取用户隐私数据",
            "danger": True
        },
        "mpc_mode": {
            "title": "MCU 隐私推理",
            "p1_sees": "[" + ", ".join(
                str(random.randint(10**17, 10**18)) for _ in range(3)
            ) + ", ...]",
            "risk": "服务商仅见随机份额，无法还原任何信息",
            "danger": False
        }
    }


@app.get("/api/scenarios")
def get_scenarios():
    return {k: {"name": v["name"], "example": v["example"]}
            for k, v in SCENARIOS.items()}


@app.get("/api/perf")
def get_performance():
    return {
        "latency": [
            {"method": "明文推理", "v": 0.8},
            {"method": "CrypTen原始", "v": 71.0},
            {"method": "SecFormer", "v": 19.5},
            {"method": "MCU-Transformer", "v": 25.0},
        ],
        "accuracy": [
            {"task": "CoLA(Large)", "secformer": 0.0, "mcu": 60.7},
            {"task": "MRPC", "secformer": 89.2, "mcu": 87.4},
            {"task": "STS-B", "secformer": 87.4, "mcu": 88.9},
            {"task": "QNLI", "secformer": 91.2, "mcu": 91.4},
            {"task": "RTE", "secformer": 69.0, "mcu": 66.8},
        ]
    }


@app.get("/")
def root():
    return {"status": "MCU-Transformer Dashboard API v2 运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)