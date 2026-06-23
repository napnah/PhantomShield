# 环境配置说明（conda base，无需新建 venv）

## 一键补齐依赖

```powershell
cd F:\AI_Agent\MCU-transformer
pip install --user -r requirements.txt
python scripts/download_models.py
```

> base 环境若 `pip install` 报权限错误，加 `--user`。  
> CrypTen 安装若遇 sklearn 报错：`$env:SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL="True"`  
> PyTorch 2.8 需 `scripts/crypten_compat.py` 补丁（实验脚本已自动加载）。

或一键配置：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

## 验证

```powershell
$env:PYTHONIOENCODING="utf-8"

# 1) Dashboard 关键词分类（后端）
cd dashboard\backend
python main.py
# 浏览器打开 dashboard\frontend\index.html

# 2) CrypTen + BERT 语义分类
cd ..\..
python experiments\crypten_bert_classify.py
```

## 当前环境快照

| 组件 | 本机版本 |
|---|---|
| Python | 3.10.20 (conda base) |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.46.0 (--user) |
| crypten | 0.4.1 (--user) |
