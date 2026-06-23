# PhantomShield 环境配置（conda base，不新建 venv）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> 安装 Python 依赖 (--user) ..."
$env:SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL = "True"
pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "==> 下载 BERT-Base ..."
python scripts/download_models.py

Write-Host "==> 验证 CrypTen ..."
python -c "import sys; sys.path.insert(0,'scripts'); from crypten_compat import patch_crypten_inprocess; patch_crypten_inprocess(); import crypten; crypten.init_thread(0,1); print('crypten OK')"

Write-Host "==> 验证 Dashboard 推理引擎 ..."
$env:PYTHONIOENCODING = "utf-8"
python -c "import sys; sys.path.insert(0,'.'); from dashboard.backend.infer_engine import full_inference; r=full_inference('患者发烧咳嗽','medical'); print(r['top_prediction'])"

Write-Host ""
Write-Host "配置完成。启动 Dashboard: cd dashboard\backend; python main.py"
Write-Host "CrypTen+BERT: python experiments\crypten_bert_classify.py"
