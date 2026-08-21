"""
Step 1 自检：检索冒烟测试
10 个探针问题，验证 top3 命中预期内容（关键词出现在 top3 块文本中）。
通过标准：命中率 >= 80%。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge_base import get_kb

PROBES = [
    ("台账授权的理论依据是什么", ["台账授权", "授权"]),
    ("浆洗街街道帮助了多少残疾人就业", ["浆洗街"]),
    ("数字悬浮概念出自哪篇文献", ["范炜烽"]),
    ("选题报告的研究方法包括哪些", ["研究方法", "访谈", "调研"]),
    ("TTF模型的因子有哪些", ["Goodhue", "授权", "兼容性"]),
    ("交易成本包括哪四种类型", ["搜寻", "协商"]),
    ("三元空间是指哪三个空间", ["物理", "社会", "数字"]),
    ("报表通系统有哪四大功能板块", ["数据看板", "我要数据", "任务中心"]),
    ("一稿的实践审思提到哪些结构性壁垒", ["结构性壁垒", "数据质量"]),
    ("社区数仓覆盖了多少个街道和社区", ["72个社区", "11个街道"]),
]


def main():
    kb = get_kb()
    hits, misses = 0, []
    for query, expect_keywords in PROBES:
        results = kb.search(query, final_k=3)
        joined = "\n".join(r["content"] for r in results)
        matched = [kw for kw in expect_keywords if kw in joined]
        hit = len(matched) >= min(2, len(expect_keywords))
        status = "HIT " if hit else "MISS"
        hits += hit
        sources = [f"{r['source']}({r['weighted_score']})" for r in results]
        print(f"[{status}] {query}")
        print(f"      命中词: {matched} | top3来源: {sources}")
        if not hit:
            misses.append(query)

    rate = hits / len(PROBES)
    print(f"\n命中率: {hits}/{len(PROBES)} = {rate:.0%} (要求 >= 80%)")
    if misses:
        print("未通过探针:", misses)
    if rate < 0.8:
        print("=== Step 1 自检未通过 ===")
        sys.exit(1)
    print("=== Step 1 自检通过 ===")


if __name__ == "__main__":
    main()
