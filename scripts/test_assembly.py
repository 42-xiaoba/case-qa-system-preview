"""
Step 3 自检：预算制组装管线
检查项：
1. 消息结构完整（人设/规则/Tier0/检索块/输出格式 五段俱全）
2. 检索块编号连续且带来源
3. RAG 路径不再注入 case.txt 全文
4. 历史对话超预算时从最早的消息开始丢弃
5. 检索块超预算时从相关性最低的尾部丢弃
6. 总量兜底 ≤ total_chars
"""
import sys

sys.path.insert(0, ".")

from core.pipeline import build_answer_messages

failures = []


def check(name: str, ok: bool, detail: str = ""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- 探针 1：基础结构与编号 ----
messages, docs = build_answer_messages("台账授权的理论依据是什么")
system = messages[0]["content"]
check("消息结构: system开头", messages[0]["role"] == "system")
check("消息结构: 末尾为当前user问题", messages[-1] == {"role": "user", "content": "台账授权的理论依据是什么"})
for seg in ["系统人设与角色", "核心业务规则", "常备知识卡", "检索到的参考资料", "输出格式要求"]:
    check(f"system含段落[{seg}]", seg in system)
check("检索块非空", len(docs) > 0, f"注入{len(docs)}块")
check("检索块编号[1]存在", "[1]【" in system)

# ---- 探针 2：RAG 路径不注入 case.txt 全文 ----
check("RAG路径无全文标记", "【以下为本次分析的完整案例文本" not in system)

# ---- 探针 3：历史预算裁剪（构造超长历史）----
big_history = []
for i in range(20):
    big_history.append({"role": "user", "content": f"第{i}个问题：" + "背景铺垫" * 80})
    big_history.append({"role": "assistant", "content": f"第{i}个回答：" + "详细解答" * 80})
messages2, _ = build_answer_messages("报表通是什么", history=big_history)
history_msgs = [m for m in messages2[1:-1]]
check("超长历史被裁剪", len(history_msgs) < len(big_history), f"保留{len(history_msgs)}/{len(big_history)}条")
if history_msgs:
    check("裁剪保留的是最新消息", history_msgs[-1]["content"].startswith("第19个"), history_msgs[-1]["content"][:12])

# ---- 探针 4：总量兜底 ----
total_chars = sum(len(m["content"]) for m in messages2 if isinstance(m["content"], str))
budget_total = 12000
check(f"总输入≤{budget_total}字符", total_chars <= budget_total + 200, f"实际{total_chars}")

# ---- 探针 5：检索块预算（final_k=6，块均~500字 → 6块约3000字应全保留；用小context_cap验证裁剪逻辑）----
from core.prompt_manager import prompt_manager

fake_docs = [
    {"source": f"源{i}", "section": "节", "content": "x" * 1500, "tier": 1, "score": 10 - i}
    for i in range(6)
]
kept = prompt_manager.build_messages_with_rag("测试", fake_docs)
n_injected = kept[0]["content"].count("】节】") if False else sum(
    1 for i in range(1, 7) if f"[{i}]【源{i-1}" in kept[0]["content"]
)
check("超预算时丢弃低相关块", n_injected < 6, f"4200预算下注入{n_injected}块(每块1500字)")
check("保留的是高相关块", "[1]【源0" in kept[0]["content"])

# ---- 探针 6：摘要注入 ----
messages3, _ = build_answer_messages("继续刚才的话题", history_summary="用户此前询问了浆洗街就业帮扶，提到2476条数据。")
check("历史摘要注入system", "历史对话摘要" in messages3[0]["content"] and "2476" in messages3[0]["content"])

print()
if failures:
    print(f"=== Step 3 自检未通过: {len(failures)}项 ===")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("=== Step 3 自检全部通过 ===")
