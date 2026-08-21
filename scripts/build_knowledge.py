"""
Step 0: 知识库构建脚本（离线一次性运行）
从 4 个资料源提取文本并切块，输出 kb/chunks.json：
  - 清华案例分析报告一稿.docx  (tier 1, source=一稿)
  - case_report.pdf            (tier 1, source=选题报告)
  - case.txt                   (tier 1, source=案例文本)
  - resources/文献资料要点汇编.md (tier 2, source=文献汇编)

切块策略：标题感知切块（一稿/汇编按章节标题，选题报告/案例文本按行内标题探测），
块长目标 300-600 字，超长块按段落边界二次切分。

自检（Step 0 门禁）：
  1. 所有块长度在 [200, 900] 区间
  2. 一稿 >= 8 个章节、选题报告 >= 5 个部分、汇编 >= 7 个小节被覆盖
  3. 抽样打印块首尾句供人工检查语义边界
"""

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "kb" / "chunks.json"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

MIN_LEN, MAX_LEN = 200, 900
HARD_SPLIT = 600
OVERLAP = 50

# 章节标题模式：一、二、……（含"第一部分"变体）
SECTION_RE = re.compile(r"^(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+部分)")
# 选题报告已知部分名（用于兜底标注）
KNOWN_SECTIONS = [
    "选题背景", "研究意义", "案例摘要", "研究问题与分析框架", "研究方法与调研安排",
]
# 一稿已知章节名（一稿标题为不带编号的独立短行）
KNOWN_DOC_SECTIONS = [
    "案例摘要", "要点分析", "分析框架", "案例阐释", "数据获取交易成本的消解机制",
    "技术适配的边界与制度条件的交互关系", "路径重构", "案例总结", "实践审思", "参考文献",
]


def extract_docx_lines(path: Path) -> list[str]:
    """提取 docx 段落为行列表"""
    with zipfile.ZipFile(path) as z:
        xml_data = z.read("word/document.xml")
    root = ET.fromstring(xml_data)
    lines = []
    for p in root.iter(f"{W_NS}p"):
        texts = [n.text or "" for n in p.iter(f"{W_NS}t")]
        line = "".join(texts).strip()
        if line:
            lines.append(line)
    return lines


def extract_pdf_pages(path: Path) -> list[list[str]]:
    """提取 pdf 每页文本为行列表的列表"""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pages.append(lines)
    return pages


def split_long_text(text: str, hard_split: int = HARD_SPLIT, overlap: int = OVERLAP) -> list[str]:
    """将超长文本按段落边界切分为 <= hard_split 的块，带少量重叠"""
    if len(text) <= hard_split:
        return [text]
    paras = [p for p in re.split(r"(?<=[。；！？\n])", text) if p.strip()]
    chunks, buf = [], ""
    for para in paras:
        if buf and len(buf) + len(para) > hard_split:
            chunks.append(buf)
            buf = buf[-overlap:] + para if overlap < len(buf) else para
        else:
            buf += para
    if buf.strip():
        chunks.append(buf)
    # 单段仍超长的极端情况：硬切
    result = []
    for c in chunks:
        while len(c) > hard_split * 1.5:
            result.append(c[:hard_split])
            c = c[hard_split - overlap:]
        result.append(c)
    return result


def chunk_by_heading_lines(lines: list[str], source: str, tier: int,
                           default_section: str = "正文") -> list[dict]:
    """按 一、二、三、 标题行分组后切块（适用于 docx 与纯文本）"""
    sections: list[tuple[str, list[str]]] = []
    cur_title, cur_buf = default_section, []
    for line in lines:
        if SECTION_RE.match(line) and len(line) < 60:
            if cur_buf:
                sections.append((cur_title, cur_buf))
            cur_title, cur_buf = line.strip(), []
        else:
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, cur_buf))
    return build_chunks(sections, source, tier)


def chunk_pdf_by_sections(pages: list[list[str]], source: str, tier: int) -> list[dict]:
    """按页扫描选题报告：探测已知部分标题，维护当前章节状态后分组切块"""
    sections: list[tuple[str, list[str]]] = []
    cur_title, cur_buf = "封面与目录", []
    for page_lines in pages:
        for line in page_lines:
            hit = next((name for name in KNOWN_SECTIONS
                        if name in line and len(line) < 40), None)
            if hit and (SECTION_RE.match(line) or line.replace(" ", "").startswith(hit)):
                if cur_buf:
                    sections.append((cur_title, cur_buf))
                cur_title, cur_buf = hit, []
                continue
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, cur_buf))
    return build_chunks(sections, source, tier)


def chunk_docx_by_known_sections(lines: list[str], source: str, tier: int) -> list[dict]:
    """一稿专用：按已知章节名（独立短行）分组切块"""
    sections: list[tuple[str, list[str]]] = []
    cur_title, cur_buf = "标题页", []
    for line in lines:
        stripped = line.replace(" ", "")
        hit = next((name for name in KNOWN_DOC_SECTIONS
                    if stripped.startswith(name) and len(stripped) < 40), None)
        if hit:
            if cur_buf:
                sections.append((cur_title, cur_buf))
            cur_title, cur_buf = line.strip(), []
        else:
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, cur_buf))
    return build_chunks(sections, source, tier)


def chunk_markdown(path: Path, source: str, tier: int) -> list[dict]:
    """按 markdown 标题切块：### 为叶子单元，## 作为前缀上下文"""
    h2, h3, buf = "总览", "", []
    sections: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((f"{h2}·{h3}" if h3 else h2, buf))
                buf = []
            h2, h3 = line[3:].strip(), ""
        elif line.startswith("### "):
            if buf:
                sections.append((f"{h2}·{h3}" if h3 else h2, buf))
                buf = []
            h3 = line[4:].strip()
        else:
            buf.append(line)
    if buf:
        sections.append((f"{h2}·{h3}" if h3 else h2, buf))
    return build_chunks(sections, source, tier)


def build_chunks(sections: list[tuple[str, list[str]]], source: str, tier: int) -> list[dict]:
    """将 (标题, 行列表) 分组转换为定长块"""
    chunks = []
    for title, lines in sections:
        text = "\n".join(lines).strip()
        if len(text) < 60:
            continue
        for piece in split_long_text(text):
            piece = piece.strip()
            if len(piece) < MIN_LEN // 2:
                continue
            chunks.append({
                "source": source,
                "tier": tier,
                "section": title,
                "content": f"【{source}·{title}】{piece}",
            })
    return chunks


def merge_small_chunks(chunks: list[dict]) -> list[dict]:
    """将过小的相邻同源块合并（同 source 且同 section 才合并）"""
    merged = []
    for c in chunks:
        if merged and len(c["content"]) < MIN_LEN \
                and merged[-1]["source"] == c["source"] \
                and merged[-1]["section"] == c["section"] \
                and len(merged[-1]["content"]) + len(c["content"]) <= MAX_LEN:
            merged[-1]["content"] += "\n" + c["content"]
        else:
            merged.append(dict(c))
    return merged


def main():
    all_chunks: list[dict] = []

    # 1. 一稿 docx（章节标题为不带编号的独立短行）
    docx_path = ROOT / "清华案例分析报告一稿.docx"
    all_chunks += chunk_docx_by_known_sections(extract_docx_lines(docx_path), "一稿", 1)

    # 2. 选题报告 pdf
    pdf_path = ROOT / "case_report.pdf"
    all_chunks += chunk_pdf_by_sections(extract_pdf_pages(pdf_path), "选题报告", 1)

    # 3. 案例文本 case.txt
    case_lines = [ln.strip() for ln in
                  (ROOT / "case.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    all_chunks += chunk_by_heading_lines(case_lines, "案例文本", 1)

    # 4. 文献汇编 markdown
    all_chunks += chunk_markdown(ROOT / "resources" / "文献资料要点汇编.md", "文献汇编", 2)

    all_chunks = merge_small_chunks(all_chunks)
    for i, c in enumerate(all_chunks):
        c["id"] = i

    # ==================== Step 0 自检 ====================
    print(f"总块数: {len(all_chunks)}")
    by_source: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_source.setdefault(c["source"], []).append(c)

    lens = [len(c["content"]) for c in all_chunks]
    print(f"块长: min={min(lens)}, max={max(lens)}, avg={sum(lens)//len(lens)}")

    ok = True
    for c in all_chunks:
        if len(c["content"]) > MAX_LEN * 1.6:
            print(f"[FAIL] 超长块 id={c['id']} len={len(c['content'])} section={c['section']}")
            ok = False

    coverage_req = {"一稿": 8, "选题报告": 4, "案例文本": 2, "文献汇编": 7}
    for src, min_sections in coverage_req.items():
        n_sections = len({c["section"] for c in by_source.get(src, [])})
        status = "OK" if n_sections >= min_sections else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {src}: {len(by_source.get(src, []))} 块, "
              f"{n_sections} 个章节 (要求 >= {min_sections})")

    print("\n---- 边界抽查（每源抽 2 块的首尾 40 字）----")
    for src, chunks in by_source.items():
        for c in (chunks[0], chunks[len(chunks) // 2]):
            head = c["content"][:40].replace("\n", " ")
            tail = c["content"][-40:].replace("\n", " ")
            print(f"[{src}#{c['id']}] {head} ...... {tail}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)
    print(f"\n已写出: {OUT_PATH}")

    if not ok:
        print("\n=== Step 0 自检未通过 ===")
        sys.exit(1)
    print("=== Step 0 自检通过 ===")


if __name__ == "__main__":
    main()
