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
) -> tuple[list[dict], list[dict]]:
    """
    构建一次问答所需的完整消息

    Args:
        user_query: 用户当前问题
        history: 历史对话 [{"role","content"},...]
        history_summary: 较早对话的滚动摘要（P1 记忆压缩产出）
        route_type: 查询类型（case/report/literature/followup），None 表示全库检索

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
    )
    return messages, docs


def build_answer_messages_routed(
    user_query: str,
    history: list | None = None,
    history_summary: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    P1 带路由的完整管线：分类（+追问改写）→ 检索 → 预算制组装

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
    )
    return messages, docs, route
