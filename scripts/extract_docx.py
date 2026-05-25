"""
从毕设.docx提取论文内容到 论文.md
- 段落文字（保留标题层级）
- 图片提取到 论文_images/ 目录
- 表格转 markdown 表格
- Word OMML 公式转 LaTeX
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DOCX_PATH = Path("md/毕设.docx")
OUTPUT_MD = Path("论文.md")
IMAGE_DIR = Path("论文_images")

# ── OMML -> MathML -> LaTeX 转换 ─────────────────────────────────────────────
OMML2MML_XSL = None

def _load_omml_xsl():
    """尝试加载 Office 自带的 OMML->MathML XSLT"""
    global OMML2MML_XSL
    candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Microsoft Office/root/Office16/OMML2MML.XSL",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Office/root/Office16/OMML2MML.XSL",
    ]
    for p in candidates:
        if p.exists():
            OMML2MML_XSL = etree.XSLT(etree.parse(str(p)))
            return True
    return False


def omml_to_latex(omml_element) -> str:
    """将 OMML XML 元素转为 LaTeX 字符串（尽力而为）"""
    if OMML2MML_XSL is None:
        return _omml_fallback(omml_element)
    try:
        mml = OMML2MML_XSL(omml_element)
        return _mml_to_latex(mml)
    except Exception:
        return _omml_fallback(omml_element)


def _omml_fallback(el) -> str:
    """无 XSL 时直接提取 OMML 中的文本"""
    texts = []
    for t in el.iter(qn("m:t")):
        if t.text:
            texts.append(t.text)
    raw = "".join(texts).strip()
    if raw:
        return f"${raw}$"
    return ""


def _mml_to_latex(mml_tree) -> str:
    """简易 MathML -> LaTeX（覆盖常见结构）"""
    root = mml_tree.getroot()
    text = etree.tostring(root, encoding="unicode")
    # 简单情况：直接取文本内容
    texts = []
    for el in root.iter():
        if el.text and el.text.strip():
            texts.append(el.text.strip())
    result = " ".join(texts)
    if result:
        return f"${result}$"
    return ""


# ── 图片提取 ─────────────────────────────────────────────────────────────────
def extract_images(docx_path: Path, out_dir: Path) -> dict[str, str]:
    """从 docx zip 中提取所有图片，返回 rId -> 相对路径 映射"""
    out_dir.mkdir(exist_ok=True)
    mapping = {}
    with ZipFile(docx_path) as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                fname = Path(name).name
                target = out_dir / fname
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                mapping[fname] = f"论文_images/{fname}"
    return mapping


# ── 段落级别检测 ──────────────────────────────────────────────────────────────
HEADING_MAP = {
    "Heading 1": "#", "Heading 2": "##", "Heading 3": "###",
    "Heading 4": "####",
    "heading 1": "#", "heading 2": "##", "heading 3": "###",
}


def get_heading_prefix(para) -> str:
    style_name = para.style.name if para.style else ""
    for key, prefix in HEADING_MAP.items():
        if key.lower() in style_name.lower():
            return prefix
    return ""


# ── 表格转换 ─────────────────────────────────────────────────────────────────
def table_to_md(table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 1:
        header_sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
        rows.insert(1, header_sep)
    return "\n".join(rows)


# ── 段落内容提取（含公式和内嵌图片）────────────────────────────────────────────
def para_to_md(para, image_mapping: dict, img_counter: list) -> str:
    """将段落转为 markdown 文本，处理行内公式和图片"""
    parts = []
    xml = para._element

    for child in xml:
        tag = child.tag
        if tag == qn("w:r"):
            drawings = child.findall(f".//{qn('a:blip')}")
            if drawings:
                for blip in drawings:
                    embed = blip.get(qn("r:embed"))
                    img_counter[0] += 1
                    parts.append(f"\n\n![图片{img_counter[0]}]({{img_{embed}}})\n\n")
            else:
                for t in child.iter(qn("w:t")):
                    if t.text:
                        parts.append(t.text)
        elif tag == qn("m:oMathPara") or tag == qn("m:oMath"):
            latex = omml_to_latex(child)
            if latex:
                if tag == qn("m:oMathPara"):
                    parts.append(f"\n\n$${latex.strip('$')}$$\n\n")
                else:
                    parts.append(latex)

    return "".join(parts)


# ── 解析 rId -> 图片文件名 映射 ───────────────────────────────────────────────
def build_rid_map(docx_path: Path) -> dict[str, str]:
    """从 word/_rels/document.xml.rels 解析 rId -> media/filename"""
    rid_map = {}
    with ZipFile(docx_path) as zf:
        rels_path = "word/_rels/document.xml.rels"
        if rels_path in zf.namelist():
            tree = etree.parse(zf.open(rels_path))
            ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
            for rel in tree.findall(".//r:Relationship", ns):
                rid = rel.get("Id")
                target = rel.get("Target", "")
                if "media/" in target:
                    fname = Path(target).name
                    rid_map[rid] = fname
    return rid_map


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main() -> None:
    _load_omml_xsl()

    print(f"[1] 提取图片到 {IMAGE_DIR}/ ...")
    img_files = extract_images(DOCX_PATH, IMAGE_DIR)
    print(f"    提取了 {len(img_files)} 个媒体文件")

    print("[2] 构建 rId -> 文件名映射 ...")
    rid_map = build_rid_map(DOCX_PATH)

    print("[3] 解析文档内容 ...")
    doc = Document(str(DOCX_PATH))

    lines: list[str] = []
    img_counter = [0]

    body = doc.element.body
    for element in body:
        tag = element.tag
        # 段落
        if tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)
            heading = get_heading_prefix(para)
            text = para_to_md(para, rid_map, img_counter)
            if heading and text.strip():
                lines.append(f"\n{heading} {text.strip()}\n")
            elif text.strip():
                lines.append(text.strip())
            else:
                lines.append("")
        # 表格
        elif tag == qn("w:tbl"):
            from docx.table import Table
            tbl = Table(element, doc)
            lines.append("")
            lines.append(table_to_md(tbl))
            lines.append("")

    # 替换图片占位符 {img_rId...} -> 实际路径
    content = "\n".join(lines)
    for rid, fname in rid_map.items():
        placeholder = f"{{img_{rid}}}"
        content = content.replace(placeholder, f"论文_images/{fname}")

    OUTPUT_MD.write_text(content, encoding="utf-8")
    print(f"[OK] 已保存到 {OUTPUT_MD} ({len(lines)} 行)")


if __name__ == "__main__":
    main()
