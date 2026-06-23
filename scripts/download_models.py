"""下载项目所需 BERT 权重（ModelScope 镜像）。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from modelscope import snapshot_download

    base = os.path.join(ROOT, "bert-base-uncased")
    if os.path.isdir(base) and os.path.isfile(os.path.join(base, "config.json")):
        print(f"[skip] bert-base-uncased 已存在: {base}")
    else:
        print("下载 bert-base-uncased ...")
        snapshot_download("AI-ModelScope/bert-base-uncased", local_dir=base)
        print(f"完成: {base}")

    large = os.path.join(ROOT, "bert-large-uncased")
    if os.path.isdir(large) and os.path.isfile(os.path.join(large, "config.json")):
        print(f"[skip] bert-large-uncased 已存在: {large}")
    else:
        print("下载 bert-large-uncased（可选，体积较大）...")
        snapshot_download("AI-ModelScope/bert-large-uncased", local_dir=large)
        print(f"完成: {large}")


if __name__ == "__main__":
    main()
