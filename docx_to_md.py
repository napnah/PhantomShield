#!/usr/bin/env python3
"""Convert MCU-Transformer docx to Markdown with LaTeX math."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CODE_FONTS = {"Courier New", "Consolas", "Courier", "Cascadia Mono", "Monaco"}

HEADING_STYLES = {
    "11": 2,
    "21": 3,
    "31": 4,
    "41": 5,
    "51": 6,
    "61": 6,
}

UNICODE_TO_LATEX = {
    "×": r"\times ",
    "⊙": r"\odot ",
    "⊗": r"\otimes ",
    "∈": r"\in ",
    "←": r"\leftarrow ",
    "≡": r"\equiv",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≠": r"\neq ",
    "∑": r"\sum ",
    "Σ": r"\sum ",
    "∏": r"\prod ",
    "∞": r"\infty ",
    "√": r"\sqrt",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "λ": r"\lambda ",
    "θ": r"\theta ",
    "σ": r"\sigma ",
    "μ": r"\mu ",
    "Π": r"\Pi ",
    "→": r"\to ",
    "⇒": r"\Rightarrow ",
    "⇔": r"\Leftrightarrow ",
    "□": r"\square",
}


def read_docx(docx_path: Path) -> ET.Element:
    with zipfile.ZipFile(docx_path) as z:
        return ET.fromstring(z.read("word/document.xml"))


def get_para_style(p: ET.Element) -> str | None:
    ppr = p.find(f"{{{W}}}pPr")
    if ppr is None:
        return None
    ps = ppr.find(f"{{{W}}}pStyle")
    return ps.get(f"{{{W}}}val") if ps is not None else None


def get_run_font(r: ET.Element) -> str | None:
    rpr = r.find(f"{{{W}}}rPr")
    if rpr is None:
        return None
    rf = rpr.find(f"{{{W}}}rFonts")
    if rf is None:
        return None
    for attr in ("ascii", "hAnsi", "eastAsia"):
        val = rf.get(f"{{{W}}}{attr}")
        if val:
            return val
    return None


def is_run_bold(r: ET.Element) -> bool:
    rpr = r.find(f"{{{W}}}rPr")
    if rpr is None:
        return False
    b = rpr.find(f"{{{W}}}b")
    return b is not None and b.get(f"{{{W}}}val", "true") != "false"


def is_run_italic(r: ET.Element) -> bool:
    rpr = r.find(f"{{{W}}}rPr")
    if rpr is None:
        return False
    i = rpr.find(f"{{{W}}}i")
    return i is not None and i.get(f"{{{W}}}val", "true") != "false"


def get_vert_align(r: ET.Element) -> str | None:
    rpr = r.find(f"{{{W}}}rPr")
    if rpr is None:
        return None
    va = rpr.find(f"{{{W}}}vertAlign")
    return va.get(f"{{{W}}}val") if va is not None else None


def run_text(r: ET.Element) -> str:
    parts: list[str] = []
    sym = r.find(f"{{{W}}}sym")
    if sym is not None:
        char = sym.get(f"{{{W}}}char")
        if char:
            parts.append(char)
    t = r.find(f"{{{W}}}t")
    if t is not None and t.text:
        parts.append(t.text)
    return "".join(parts)


def extract_runs(p: ET.Element) -> list[dict]:
    runs: list[dict] = []
    for r in p.findall(f"{{{W}}}r"):
        text = run_text(r)
        if not text:
            continue
        runs.append(
            {
                "text": text,
                "bold": is_run_bold(r),
                "italic": is_run_italic(r),
                "font": get_run_font(r),
                "vert": get_vert_align(r),
            }
        )
    return runs


def runs_to_plain(runs: list[dict]) -> str:
    return "".join(r["text"] for r in runs)


def is_code_paragraph(runs: list[dict]) -> bool:
    if not runs:
        return False
    fonts = {r["font"] for r in runs if r["font"]}
    return bool(fonts & CODE_FONTS)


def paragraph_plain(p: ET.Element) -> str:
    return runs_to_plain(extract_runs(p))


def replace_unicode_math(text: str) -> str:
    for ch, latex in UNICODE_TO_LATEX.items():
        text = text.replace(ch, latex)
    # Math dot product only in formula-like contexts.
    text = re.sub(
        r"(?<=[A-Za-z0-9_\]\)\}])\s*·\s*(?=[A-Za-z0-9_\[\(\{])",
        r" \\cdot ",
        text,
    )
    text = re.sub(r"\\sqrt([a-zA-Z0-9_]+)", r"\\sqrt{\1}", text)
    text = re.sub(r"\(mod\s+([A-Za-z0-9_\^]+)\)", r"\\pmod{\1}", text)
    text = re.sub(r"\bmod\s+([A-Za-z0-9_\^]+)\b", r"\\pmod{\1}", text)
    text = re.sub(r"\\Pi\s+_", r"\\Pi_", text)
    text = re.sub(r"\^(\\Pi\b)", r"^{\1}", text)
    return text


MATH_LINE_RE = re.compile(
    r"^[\s{]*"
    r"(?:\[\[?[xXyYzZeEwW][^\]]*\]\]?|[A-Za-z_][A-Za-z0-9_]*\s*=|"
    r"\{[^}]+\}|\\[a-zA-Z]+|[∑∏√≡∈·×⊙⊗≤≥←])"
)

MATH_HEAVY_RE = re.compile(
    r"(=|\[\[|\]\]|_\{|_\w|\^\{|\\\w+|\\pmod|\bmod\b|"
    r"\\cdot|\\times|\\odot|\\sqrt|\\equiv|\\Pi|\\sum|\\prod|\\leftarrow)"
)

INLINE_MATH_RE = re.compile(
    r"\[\[[^\]]+\]\]_[a-zA-Z0-9]+|"
    r"\[\[[^\]]+\]\]|"
    r"\[[^\]]+\]_[a-zA-Z0-9]+|\[[^\]]+\]|"
    r"e\^\([^)]+\)|"
    r"[A-Za-z_][A-Za-z0-9_]*\^[\{\w][^\s，。；：、）)]*|"
    r"[A-Za-z_][A-Za-z0-9_]*_[a-zA-Z0-9]+|"
    r"\\Pi_[a-zA-Z]+|"
    r"softmax\([^)]+\)|"
    r"GeLU\([^)]+\)|"
    r"sigmoid\([^)]+\)|"
    r"wrap\([^)]+\)|"
    r"Share\([^)]+\)|"
    r"Recon\([^)]+\)|"
    r"View_[A-Za-z]+\^[^\s]+|"
    r"Sim_[A-Za-z]+\([^)]+\)|"
    r"Z_L|2\^64|2\*\*64|"
    r"QK\^T|"
    r"1\.702\s*[·\\cdot]\s*\[x\]_i|"
    r"1\[x\+r\s*[≤\\leq]\s*2\^l\]|"
    r"\\pmod\{[A-Za-z0-9_\^]+\}"
)


def is_display_math_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if re.match(r"^[\-\*]\s", stripped):
        return False
    if re.match(r"^\d+\.\s+[\u4e00-\u9fff]", stripped):
        return False
    if re.match(r"^(定义|定理|证明|步骤|其中|输入|输出|统一公式|核心恒等式|线性操作|Share|Recon)", stripped):
        return False
    if "：" in stripped and re.search(r"[\u4e00-\u9fff]", stripped):
        return False
    if re.search(r"\b(def|class|import|return|self\.)\b", stripped):
        return False
    if len(stripped) > 220:
        return False
    if stripped.endswith("□"):
        stripped = stripped[:-1].strip()
    chinese = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    if chinese > 12:
        return False
    if MATH_LINE_RE.match(stripped):
        return True
    if "=" in stripped and MATH_HEAVY_RE.search(stripped):
        return chinese <= 6
    return False


def wrap_inline_math(text: str) -> str:
    text = re.sub(r"(?<!\$)\\Pi(?!\w)", r"$\\Pi$", text)
    text = re.sub(r"(?<!\$)Π(?!\w)", r"$\\Pi$", text)
    parts: list[str] = []
    last = 0
    for m in INLINE_MATH_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()])
        seg = replace_unicode_math(m.group())
        parts.append(f"${seg}$")
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return "".join(parts)


def format_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if is_display_math_line(text):
        body = replace_unicode_math(text.rstrip("□").strip())
        suffix = " $\\square$" if text.rstrip().endswith("□") else ""
        return f"$${body}$${suffix}"
    return wrap_inline_math(replace_unicode_math(text))


def format_paragraph(text: str, style: str | None) -> str:
    text = text.rstrip()
    if not text.strip():
        return ""
    if style == "12":
        return f"- {format_text(text)}"
    level = HEADING_STYLES.get(style or "")
    if level:
        return f"{'#' * level} {text.strip()}"
    return format_text(text)


def cell_paragraphs(cell: ET.Element) -> list[str]:
    lines: list[str] = []
    for p in cell.findall(f".//{{{W}}}p"):
        plain = paragraph_plain(p).strip()
        if plain:
            lines.append(plain)
    return lines


def table_shape(tbl: ET.Element) -> tuple[int, int]:
    rows = tbl.findall(f"{{{W}}}tr")
    if not rows:
        return 0, 0
    col_counts = [len(row.findall(f"{{{W}}}tc")) for row in rows]
    return len(rows), max(col_counts)


def is_code_text(text: str) -> bool:
    markers = (
        "def ",
        "class ",
        "import ",
        "return ",
        "self.",
        "python ",
        "# ",
        "L = 2**64",
        "m0 = ",
        "m1 = ",
    )
    return any(m in text for m in markers)


def render_single_cell_box(tbl: ET.Element) -> str:
    cell = tbl.findall(f".//{{{W}}}tc")[0]
    paras = cell_paragraphs(cell)
    joined = "\n".join(paras)
    if is_code_text(joined):
        return "```python\n" + joined + "\n```"
    return "\n\n".join(format_text(p) for p in paras)


def render_table(tbl: ET.Element) -> str:
    rows = tbl.findall(f"{{{W}}}tr")
    matrix: list[list[list[str]]] = []
    for tr in rows:
        row: list[list[str]] = []
        for tc in tr.findall(f"{{{W}}}tc"):
            row.append(cell_paragraphs(tc))
        if row:
            matrix.append(row)

    if not matrix:
        return ""

    row_count, col_count = table_shape(tbl)
    if row_count == 1 and col_count == 1:
        return render_single_cell_box(tbl)

    width = max(len(r) for r in matrix)

    def cell_md(paras: list[str]) -> str:
        if not paras:
            return ""
        if len(paras) == 1:
            return format_text(paras[0]).replace("\n", " ")
        return "<br>".join(format_text(p) for p in paras)

    norm = [r + [[] for _ in range(width - len(r))] for r in matrix]
    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []

    def esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    lines = [
        "| " + " | ".join(esc(cell_md(c)) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(esc(cell_md(c)) for c in row) + " |")
    return "\n".join(lines)


def convert_docx(docx_path: Path) -> str:
    root = read_docx(docx_path)
    body = root.find(f".//{{{W}}}body")
    if body is None:
        raise ValueError("No document body found")

    md_lines: list[str] = []
    code_buffer: list[str] = []

    def flush_code() -> None:
        nonlocal code_buffer
        if not code_buffer:
            return
        md_lines.append("```")
        md_lines.extend(code_buffer)
        md_lines.append("```")
        md_lines.append("")
        code_buffer = []

    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "sectPr":
            continue

        if tag == "tbl":
            flush_code()
            table_md = render_table(child)
            if table_md:
                md_lines.append(table_md)
                md_lines.append("")
            continue

        if tag != "p":
            continue

        runs = extract_runs(child)
        plain = runs_to_plain(runs)
        if not plain.strip():
            flush_code()
            md_lines.append("")
            continue

        if is_code_paragraph(runs):
            code_buffer.append(plain.rstrip())
            continue

        flush_code()
        formatted = format_paragraph(plain, get_para_style(child))
        if formatted:
            md_lines.append(formatted)
            md_lines.append("")

    flush_code()
    text = "\n".join(md_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    docx_files = list(Path(".").glob("*.docx"))
    if not docx_files:
        raise SystemExit("No .docx file found")
    docx_path = docx_files[0]
    md = convert_docx(docx_path)

    out_path = Path("MCU-Transformer_完整技术方案.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"Converted: {docx_path.name}")
    print(f"Output:    {out_path.name}")
    print(f"Lines:     {len(md.splitlines())}")


if __name__ == "__main__":
    main()
