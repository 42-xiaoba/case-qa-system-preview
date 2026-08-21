"""
P1 自检：查询路由器
15 道分类题，有效准确率 >= 90% 视为通过。
判定对齐 core/pipeline._ROUTE_TIER 的真实检索范围：
- report/case 同为 tier1 范围，二者混淆不影响检索 → 视为等价类
- literature(tier2) / followup(全库) 跨等价类混淆才算错误
快速通道题（短且无指代词）单独验证不触发 LLM 调用。
"""
import sys

sys.path.insert(0, ".")

from core.router import classify_query

CASES = [
    # (问题, 期望类型)
    ("这个案例的分析框架是什么", "report"),
    ("一稿的核心结论是什么", "report"),
    ("研究方法包括哪些", "report"),
    ("实践审思提到的结构性壁垒有哪些", "report"),
    ("选题报告的研究意义是什么", "report"),
    ("社区数仓覆盖了多少个街道和社区", "case"),
    ("报表通系统有哪四大功能板块", "case"),
    ("浆洗街街道帮助了多少残疾人就业", "case"),
    ("报表数量压减了百分之多少", "case"),
    ("数字悬浮概念出自哪篇文献", "literature"),
    ("TTF模型是谁提出来的", "literature"),
    ("交易成本理论的核心观点是什么", "literature"),
    ("那它的作用是什么", "followup"),
    ("这个框架怎么应用到实践中", "followup"),
    ("继续详细说说该机制", "followup"),
]

# 与 _ROUTE_TIER 对齐的等价类：同组内可互换
EQUIV = {
    "report": {"report", "case"},
    "case": {"report", "case"},
    "literature": {"literature"},
    "followup": {"followup"},
}

failures = []
correct = 0
strict_correct = 0
for q, expect in CASES:
    route = classify_query(q)
    got = route["type"] or "all"
    ok = got in EQUIV[expect]
    strict_correct += got == expect
    correct += ok
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] 期望={expect:10s} 实际={got:10s} 改写={route['rewritten_query'][:30]} | {q}")
    if not ok:
        failures.append((q, expect, got))

# 快速通道验证：短问题不应触发 LLM
fast = classify_query("报表通是什么")
print(f"\n快速通道: type={fast['type']}, fast_path={fast['fast_path']} (期望 type=None, fast_path=True)")
fast_ok = fast["type"] is None and fast["fast_path"]

n = len(CASES)
rate = correct / n
print(f"\n有效准确率(等价类): {correct}/{n} = {rate:.0%} | 严格准确率(原始标签): {strict_correct}/{n} (要求有效 >= 90%)")
if rate >= 0.9 and fast_ok and not failures:
    print("=== 路由器自检通过 ===")
else:
    print(f"=== 路由器自检未通过 === 快速通道={'OK' if fast_ok else 'FAIL'}")
    for q, e, g in failures:
        print(f" - {q}: 期望{e} 实际{g}")
    sys.exit(1)
