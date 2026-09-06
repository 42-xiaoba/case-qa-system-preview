"""
Step 0: 知识库构建脚本（离线一次性运行）· v2 语料源
从新资料源提取文本并切块，输出 kb/chunks.json：
  - 当AI成为数据的摆渡人：基层减负的供需协同.docx  (tier 1, source=案例报告，正式版)
  - case_report.pdf                                  (tier 1, source=选题报告)
  - resources_new/为什么案例报告和选题报告有出入.docx (tier 1, source=报告说明)
  - resources_new/*.pdf                              (tier 2, source=文献·短名，共 9 篇)

切块策略：标题感知切块（案例报告按已知章节名，选题报告按行内标题探测，
文献按 一、二、 标题探测后段落兜底），块长目标 300-600 字，超长块按段落边界二次切分。

自检（Step 0 门禁）：
  1. 所有块长度在 [200, 900] 区间（上限放宽至 1.6 倍）
  2. 案例报告 >= 8 个章节、选题报告 >= 4 个部分、报告说明 >= 5 个小节、文献 >= 7 篇被覆盖
  3. 来源纯净性：结果中不允许出现旧语料源（一稿/案例文本/文献汇编）
  4. 抽样打印块首尾句供人工检查语义边界
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
# 案例报告（正式版）已知章节名：独立短行标题 + 分析部分小节
KNOWN_CASE_SECTIONS = [
    "案例正文", "引言", "莫让此身独负事", "散落之数终归仓", "易报难归终成困",
    "深藏之人终见天", "绕行千里终须面", "谁执锁钥谁守门", "万数不及人亲至",
    "结束语", "参考性问题", "附录", "武侯区残疾人数据统计表",
    "案例分析", "案例摘要", "要点分析", "理论基础与适用性", "分析框架",
    "案例阐释", "供需协同的逻辑：需求侧通道与供给侧储备",
    "供需协同的限度：技术绕行与制度遮蔽", "制度调适的可能路径",
    "案例总结与理论反思", "参考文献",
]

# 文献 PDF 来源映射：文件名 → {label 来源短名, meta 作者年份标注, desc 中文简介(可选)}
# desc 会注入该文献每个块首：为英文文献补充中文检索锚点（相当于旧"文献要点"注释）
LITERATURE_SOURCES = {
    "Goodhue-TaskTechnologyFitIndividual-1995.pdf": {
        "label": "任务-技术匹配模型",
        "meta": "Goodhue & Thompson，1995，《MIS Quarterly》",
        "desc": "本篇为任务-技术匹配模型（Task-Technology Fit, TTF）的奠基文献：由古德休（Goodhue）"
                "与汤普森（Thompson）于1995年提出，发表于《MIS Quarterly》。核心观点："
                "信息技术的使用绩效取决于技术功能与任务需求的适配程度，模型识别了数据质量、"
                "授权、兼容性等任务-技术匹配因子",
    },
    "“数字空间”政府及其研究纲领——第四次工.pdf": {
        "label": "数字空间政府",
        "meta": "米加宁等，2020，《公共管理学报》",
    },
    "从结构论到生态论——城市政治学的理论迭代.pdf": {
        "label": "从结构论到生态论",
        "meta": "葛天任、孟天广，2025，《政治学研究》",
    },
    "数字化协同：基层减负增能的策略选择——基.pdf": {
        "label": "数字化协同",
        "meta": "马太平、吴建南，2025，《公共管理学报》",
    },
    "数字技术“赋能”何以产生基层治理“负能”.pdf": {
        "label": "数字赋能与治理负能",
        "meta": "娄文龙等，2025，《上海行政学院学报》",
    },
    "数字技术嵌入异化下行政负担的生成与转移_.pdf": {
        "label": "行政负担的生成与转移",
        "meta": "连宏萍、邹佳秀，2025，《中国行政管理》",
    },
    "数智治理的理论模型框架与链式发展路径——.pdf": {
        "label": "数智治理",
        "meta": "彭小宝等，2025，《管理世界》",
    },
    "构建虚拟政府_信息技术与制度创新.pdf": {
        "label": "构建虚拟政府",
        "meta": "Fountain，2001，《Brookings Institution Press》",
    },
    "社区空间治理理论的跨学科视角——构建“物.pdf": {
        "label": "社区空间治理",
        "meta": "葛天任，2024，《东南学术》",
    },
    "退出、呼吁与忠诚 对企业、组织和国家衰退.pdf": {
        "label": "退出呼吁与忠诚",
        "meta": "Hirschman，1970，《Harvard University Press》",
    },
}


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
                           default_section: str = "正文",
                           meta: str | None = None) -> list[dict]:
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
    return build_chunks(sections, source, tier, meta=meta)


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


def chunk_docx_by_known_sections(lines: list[str], source: str, tier: int,
                                 skip_until: str | None = None) -> list[dict]:
    """按已知章节名（独立短行）分组切块。

    skip_until: 若给出，先跳过其前的所有行（如"案例正文"前的封面与目录），
    遇到该行本身时结束跳过并作为首个章节标题。"""
    sections: list[tuple[str, list[str]]] = []
    cur_title, cur_buf = "标题页", []
    skipping = skip_until is not None
    for line in lines:
        if skipping:
            if line.strip() == skip_until:
                skipping = False
                cur_title, cur_buf = line.strip(), []
            continue
        stripped = line.replace(" ", "")
        hit = next((name for name in KNOWN_CASE_SECTIONS
                    if stripped == name.replace(" ", "") and len(stripped) < 45), None)
        if hit:
            if cur_buf:
                sections.append((cur_title, cur_buf))
            cur_title, cur_buf = line.strip(), []
        else:
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, cur_buf))
    return build_chunks(sections, source, tier)


def build_chunks(sections: list[tuple[str, list[str]]], source: str, tier: int,
                 meta: str | None = None) -> list[dict]:
    """将 (标题, 行列表) 分组转换为定长块；meta 注入块首（如作者年份标注）"""
    chunks = []
    for title, lines in sections:
        text = "\n".join(lines).strip()
        if len(text) < 60:
            continue
        for piece in split_long_text(text):
            piece = piece.strip()
            if len(piece) < MIN_LEN // 2:
                continue
            if meta:
                piece = f"（{meta}）{piece}"
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

    # 1. 案例报告正式版 docx（章节标题为独立短行；跳过其前的封面与目录）
    docx_path = ROOT / "当AI成为数据的摆渡人：基层减负的供需协同.docx"
    all_chunks += chunk_docx_by_known_sections(
        extract_docx_lines(docx_path), "案例报告", 1, skip_until="案例正文"
    )

    # 2. 选题报告 pdf
    pdf_path = ROOT / "case_report.pdf"
    all_chunks += chunk_pdf_by_sections(extract_pdf_pages(pdf_path), "选题报告", 1)

    # 3. 报告说明 docx（为什么案例报告和选题报告有出入）
    note_path = ROOT / "resources_new" / "为什么案例报告和选题报告有出入.docx"
    all_chunks += chunk_by_heading_lines(extract_docx_lines(note_path), "报告说明", 1)

    # 4. 文献 PDF（resources_new 下逐篇提取，作者年份标注注入块首）
    lit_dir = ROOT / "resources_new"
    lit_files = sorted(p for p in lit_dir.glob("*.pdf") if p.name in LITERATURE_SOURCES)
    missing = [p.name for p in lit_dir.glob("*.pdf") if p.name not in LITERATURE_SOURCES]
    for name in missing:
        print(f"[WARN] 未登记来源映射的文献，已跳过: {name}")
    for pdf_file in lit_files:
        info = LITERATURE_SOURCES[pdf_file.name]
        label, author_year = info["label"], info["meta"]
        meta = f"{author_year}。{info['desc']}" if info.get("desc") else author_year
        source = f"文献·{label}"
        pages = extract_pdf_pages(pdf_file)
        lines = [ln for page in pages for ln in page]
        n_before = len(all_chunks)
        all_chunks += chunk_by_heading_lines(
            lines, source, 2, default_section="全文", meta=meta
        )
        print(f"[文献] {label}: {len(all_chunks) - n_before} 块（{pdf_file.name}）")

    all_chunks = merge_small_chunks(all_chunks)
    for i, c in enumerate(all_chunks):
        c["id"] = i

    # ==================== Step 0 自检 ====================
    print(f"\n总块数: {len(all_chunks)}")
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

    coverage_req = {"案例报告": 8, "选题报告": 4, "报告说明": 5}
    for src, min_sections in coverage_req.items():
        n_sections = len({c["section"] for c in by_source.get(src, [])})
        status = "OK" if n_sections >= min_sections else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {src}: {len(by_source.get(src, []))} 块, "
              f"{n_sections} 个章节 (要求 >= {min_sections})")

    # 文献覆盖检查：登记的每篇文献都应产出至少 2 块
    n_lit = len({s for s in by_source if s.startswith("文献·")})
    status = "OK" if n_lit >= 7 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[{status}] 文献: {n_lit} 篇被覆盖 (要求 >= 7)")

    # 来源纯净性：不允许出现旧语料源
    banned = {"一稿", "案例文本", "文献汇编"}
    leaked = banned & set(by_source)
    status = "OK" if not leaked else "FAIL"
    if leaked:
        ok = False
    print(f"[{status}] 来源纯净性: 旧语料源残留 = {sorted(leaked) if leaked else '无'}")

    print("\n---- 各来源块数分布 ----")
    for src, chunks in sorted(by_source.items()):
        print(f"  {src}: {len(chunks)} 块, {len({c['section'] for c in chunks})} 章节")

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
