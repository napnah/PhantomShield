"""Patch dashboard/frontend/index.html for BERT three-path UI."""
from pathlib import Path
import re

p = Path(__file__).resolve().parents[1] / "dashboard" / "frontend" / "index.html"
text = p.read_text(encoding="utf-8")

if "bertModeRow" not in text:
    insert = """<button id="inferBtn" onclick="runInfer()">加密推理</button>
      </div>
      <div class="infer-row" id="bertModeRow" style="display:none;margin-top:12px;align-items:center">
        <span style="color:var(--ink-dim);font-size:14px;margin-right:8px">BERT mode:</span>
        <select id="bertMode" style="background:var(--bg);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:15px">
          <option value="plaintext">Plaintext</option>
          <option value="crypten">CrypTen 12L+2Quad</option>
          <option value="mcu_rust" selected>MCU-Rust 12L</option>
        </select>
        <span id="methodTag" style="margin-left:12px;font-family:var(--mono);font-size:13px;color:var(--key)"></span>
      </div>"""
    text = re.sub(
        r'<button id="inferBtn" onclick="runInfer\(\)">[^<]*</button>\s*</div>',
        insert,
        text,
        count=1,
    )

text = re.sub(
    r"sentiment: \{ title: '[^']*', example: '[^']*' \}",
    "sentiment: { title: '舆情情感分析 (BERT English)', example: 'This movie is wonderful and heartwarming.' }",
    text,
    count=1,
)

if "async function loadPerf" not in text:
    text = text.replace("const PERF = {", "let PERF = {")
    perf_loader = """
async function loadPerf() {
  try {
    const res = await fetch(API + '/api/perf');
    const data = await res.json();
    if (data.latency) {
      PERF.latency = data.latency.map((d, i) => ({
        method: d.method, v: d.v,
        color: ['#34e08a', '#ff4d57', '#4d9fff', '#ffa040'][i % 4],
      }));
    }
    if (data.accuracy && data.accuracy.length) {
      const a = data.accuracy[0];
      PERF.acc = [{
        task: a.task || 'SST-2 mini',
        sec: (a.crypten || 0).toFixed(1),
        mcu: (a.mcu || 0).toFixed(1),
        win: (a.mcu || 0) >= (a.crypten || 0),
      }];
    }
    renderLatency();
    renderAcc();
  } catch (e) {
    renderLatency();
    renderAcc();
  }
}
"""
    text = text.replace("function switchScen(scen) {", perf_loader + "\nfunction switchScen(scen) {")

if "getElementById('bertModeRow')" not in text:
    text = text.replace(
        "document.getElementById('leakCipher').textContent = '[等待推理...]';",
        "document.getElementById('leakCipher').textContent = '[等待推理...]';\n"
        "  document.getElementById('bertModeRow').style.display = (scen === 'sentiment') ? 'flex' : 'none';",
    )

if "/api/infer/semantic" not in text:
    old = (
        "const res = await fetch(API + '/api/infer', { method: 'POST', "
        "headers: {'Content-Type':'application/json'}, "
        "body: JSON.stringify({ text, scenario: currentScen }) });"
    )
    new = """let res;
    if (currentScen === 'sentiment') {
      const mode = document.getElementById('bertMode').value;
      res = await fetch(API + '/api/infer/semantic', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ text, mode, max_seq_len: 32 }) });
    } else {
      res = await fetch(API + '/api/infer', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ text, scenario: currentScen }) });
    }"""
    text = text.replace(old, new)
    text = text.replace(
        "document.getElementById('predLabel').textContent = data.top_prediction;",
        "document.getElementById('predLabel').textContent = data.top_prediction || data.label;",
    )
    text = text.replace(
        "document.getElementById('predMeta').textContent = `${data.elapsed_seconds}s · ${data.comm_rounds}轮通信`;",
        "document.getElementById('predMeta').textContent = data.method ? (data.method + ' · ' + data.elapsed_seconds + 's') : (`${data.elapsed_seconds}s · ${data.comm_rounds}轮通信`);\n"
        "    if (data.method) document.getElementById('methodTag').textContent = data.method;",
    )

text = text.replace("matrixRain(); renderLatency(); renderAcc();", "matrixRain(); loadPerf();")

p.write_text(text, encoding="utf-8")
print("patched ok", p)
