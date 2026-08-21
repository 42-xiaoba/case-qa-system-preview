"""
P1 自检：记忆压缩
1. 窗口裁剪：超阈值历史被压缩为 recent_turns*2 条，摘要非空
2. 长程召回：第2轮埋入的事实（31家企业），在第13轮追问时仍能答对
"""
import sys
import time

sys.path.insert(0, ".")

from core.config import settings
from core.llm_client import llm_client
from core.memory import prepare
from core.pipeline import build_answer_messages

failures = []


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- 构造 16 轮对话（32条消息），第1轮埋入关键事实 ----
# 按生产环境的真实消息长度构造（每轮回答一两百字以上）：
# 若用超短消息凑数，保留窗口本身就占原历史的大头，压缩收益无法体现，断言必然失真。
history = [
    {"role": "user", "content": "浆洗街街道的就业帮扶是怎么做的？"},
    {"role": "assistant", "content": (
        "浆洗街街道依托社区数仓的数据回流机制开展重点群体就业帮扶："
        "先由数仓从2476条回流数据中比对测算出有用工需求的企业共31家，"
        "再经社区网格员逐户核实形成230条有效求职线索，"
        "最后由街道统一对接企业岗位，最终帮助15名居民实现就业。"
        "整个流程把过去依赖人工摸排的被动模式，转变为数据驱动的主动匹配。"
    )},
]
filler_answers = [
    "报表通的需求侧设计围绕数据看板、我要数据、数据采集、任务中心四大板块展开。数据看板面向领导驾驶舱场景，动态展示核心指标运行态势；我要数据支持基层干部按权限自助申请所需数据，改变了以往层层发函要数的做法；数据采集将原本分散在多张纸质报表中的指标统一为在线表单，从源头杜绝重复填报；任务中心对报表任务进行全流程跟踪与催办。四个板块环环相扣，共同实现了报表生产的在线化、标准化与可追溯。",
    "交易成本理论认为，组织在获取与使用数据的过程中会产生搜寻、协商、执行、核实四类成本。报表通上线之前，社区干部为了核对一张报表的口径，往往要在多个部门之间反复打电话、发函件，对应的是高昂的搜寻成本与核实成本；台账授权机制通过一次性明确各部门的数据责任边界并完成授权登记，使后续的数据调用无需重复协商，主要消解的正是搜寻与核实这两类成本，也被一稿视为最具推广价值的制度创新之一。",
    "社区数仓覆盖武侯区11个街道72个社区，汇聚了5000多个数据源，建成人口、房屋、事件、企业等基础主题库。数仓通过统一的数据标准和定时回流机制，把原本沉淀在各条线业务系统里、彼此割裂的数据整合为基层可以直接调用的资源池，为报表自动化提供了坚实的数据底座。可以说，没有数仓的汇聚治理，前端任何智能化应用都是无源之水，这也是本案例区别于一般信息化项目的关键所在。",
    "技术执行框架强调技术并非中性工具，其实际效果取决于所嵌入的制度环境；与此同时，组织形式和制度安排也会反过来对技术进行过滤和形塑。报表通在科层体制下的落地过程正是制度与技术互构的典型样本：一方面，平台重塑了报表生产的流程与分工，使数据责任更加清晰；另一方面，考核惯性与部门壁垒也在不断修正平台的使用方式，倒逼平台持续迭代，这种双向作用为理解数字政务提供了更立体的分析视角。",
    "任务-技术匹配模型关注技术功能与任务需求之间的匹配程度，八项匹配因子中兼容性与授权机制是本案例的关键：兼容性决定了平台能否顺畅嵌入基层既有的工作习惯与考核体系，授权机制决定了数据能否在合规前提下高效流动。两者共同解释了为什么同一套平台在不同社区会产出差异明显的使用效果，也为后续推广时的差异化配置提供了理论依据。",
    "一稿提出供需协同的三重路径：其一是数据归集消解获取成本，让基层一次授权、多次复用，避免向多头重复索要；其二是智能填充消解执行成本，把干部从机械重复的录入工作中解放出来；其三是反向赋能重构服务流程，使数据回流能够支撑主动发现、精准画像与前瞻决策。三条路径由浅入深、层层递进，构成了理解整个案例价值的主线框架，也回应了基层减负与服务能力提升的双重诉求。",
]
for i in range(2, 16):  # 第2~15轮铺垫问答
    history.append({"role": "user", "content": f"再展开讲讲第{i}个方面吧"})
    history.append({"role": "assistant", "content": filler_answers[i % len(filler_answers)]})
history.append({"role": "user", "content": "好的，我大概理解了"})
history.append({"role": "assistant", "content": "如果还有具体问题，欢迎继续提问。"})

threshold = int(settings.memory_config.get("compress_threshold", 14))
keep_n = int(settings.memory_config.get("recent_turns", 6)) * 2
check(f"构造的历史超过压缩阈值({threshold})", len(history) > threshold, f"{len(history)}条")

# ---- 执行压缩 ----
t0 = time.time()
windowed, summary = prepare(history, None)
latency = round(time.time() - t0, 1)
check(f"窗口裁剪为最近{keep_n}条", len(windowed) == keep_n, f"实际{len(windowed)}条")
check("滚动摘要非空", bool(summary.strip()), f"{len(summary)}字 {latency}s")
orig_chars = sum(len(m["content"]) for m in history)
new_chars = sum(len(m["content"]) for m in windowed) + len(summary)
ratio = new_chars / orig_chars
check("压缩后总量显著减小(<70%)", ratio < 0.7, f"{new_chars}/{orig_chars} ({ratio:.0%})")

# ---- 长程召回：窗口外的第2轮事实能否通过摘要找回 ----
messages, docs = build_answer_messages(
    "最开始提到的那次就业帮扶一共对接了多少家企业？",
    history=windowed,
    history_summary=summary,
)
answer = llm_client.chat(messages, temperature=0.2, max_tokens=512)
recall_ok = "31" in answer
check("长程召回: 摘要中的'31家企业'可被答出", recall_ok, answer[:80].replace("\n", " "))

print()
if failures:
    print(f"=== 记忆压缩自检未通过: {len(failures)}项 ===")
    sys.exit(1)
print("=== 记忆压缩自检全部通过 ===")
