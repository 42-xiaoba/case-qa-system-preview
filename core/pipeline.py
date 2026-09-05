"""
问答管线（Step 3 + P1 路由）
检索 → 预算制组装 的统一入口，供 ui.py 直连模式与 app.py API 模式共用。

- build_answer_messages: 纯组装入口（route_type 显式传入，None=全库检索）
- build_answer_messages_routed: P1 带路由入口（自动分类+追问改写）
"""

from core.knowledge_base import get_kb
from core.prompt_manager import prompt_manager
from core.router import classify_query

# route_type → tier_filter 映射
# report/case 类问题只查自有文档；literature 类问题只查文献汇编
_ROUTE_TIER: dict[str, list[int]] = {
    "report": [1],
    "case": [1],
    "literature": [2],
    # followup → None 全库检索（追问语境需要全部资料）
}


def build_answer_messages(
    user_query: str,
    history: list | None = None,
    history_summary: str | None = None,
    route_type: str | None = None,
    perspective: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    构建一次问答所需的完整消息

    Args:
        user_query: 用户当前问题
        history: 历史对话 [{"role","content"},...]
        history_summary: 较早对话的滚动摘要（P1 记忆压缩产出）
        route_type: 查询类型（case/report/literature/followup），None 表示全库检索
        perspective: 回答视角 key（citizen/grassroots/data_officer/director），None = 默认视角

    Returns:
        (messages, docs)：可直接发给 LLM 的消息列表 + 本次实际注入的检索块
    """
    kb = get_kb()
    tier_filter = _ROUTE_TIER.get(route_type) if route_type else None
    docs = kb.search(user_query, tier_filter=tier_filter)
    messages = prompt_manager.build_messages_with_rag(
        user_query,
        docs,
        history=history,
        history_summary=history_summary,
        perspective=perspective,
    )
    return messages, docs


def build_answer_messages_routed(
    user_query: str,
    history: list | None = None,
    history_summary: str | None = None,
    perspective: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    P1 带路由的完整管线：分类（+追问改写）→ 检索 → 预算制组装

    Args:
        perspective: 回答视角 key（None = 默认视角）

    Returns:
        (messages, docs, route_info)
    """
    route = classify_query(user_query, history)
    rewritten = route["rewritten_query"]
    messages, docs = build_answer_messages(
        rewritten,
        history=history,
        history_summary=history_summary,
        route_type=route["type"],
        perspective=perspective,
    )
    return messages, docs, route


# ==================== 延伸问题随答输出 ====================
FOLLOWUP_TAG = "<followups>"
_FOLLOWUP_INSTRUCTION = (
    "\n\n【输出格式硬性要求】回答正文结束后，必须另起一行追加延伸问题块，"
    "格式严格为：\n"
    '<followups>{"related":"紧扣本次回答内容的延伸追问","new":"切换到相关新角度的话题问题"}'
    "</followups>\n"
    "两个问题各不超过25个字，独立成句、不使用指代词；"
    "该块不属于正文，正文中不得提及它的存在。"
)


def append_followup_instruction(messages: list[dict]) -> list[dict]:
    """把延伸问题输出指令拼入最后一条用户消息（返回新列表，不改原列表）。

    拼进用户消息而非追加独立的尾部 system 消息：实测 GLM 对消息序列
    末尾的 system 角色关注度不稳定，随答标记约一半概率被忽略；
    用户消息末尾是注意力最强的位置，指令遵循率显著更高。
    兼容多模态消息（content 为分段列表）的情形。"""
    if not messages or messages[-1].get("role") != "user":
        return messages + [{"role": "system", "content": _FOLLOWUP_INSTRUCTION}]
    out = [dict(m) for m in messages]
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, list):
        parts = [dict(p) for p in content]
        for p in parts:
            if p.get("type") == "text":
                p["text"] = p.get("text", "") + _FOLLOWUP_INSTRUCTION
                break
        last["content"] = parts
    else:
        last["content"] = (content or "") + _FOLLOWUP_INSTRUCTION
    out[-1] = last
    return out
