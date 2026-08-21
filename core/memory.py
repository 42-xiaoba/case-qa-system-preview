"""
对话记忆管理（P1）
滑动窗口 + 滚动摘要，解决长对话上下文溢出与早期信息丢失的矛盾：
- 最近 recent_turns 轮消息原样保留（进入 history）
- 更早的消息压缩为滚动摘要（进入 system prompt 的【历史对话摘要】段）
- 摘要合并旧摘要滚动更新，关键实体与数据必须保留

prepare() 返回 (windowed_history, new_summary)：
- 消息数未超阈值 → 原样返回，不触发 LLM 调用（零开销）
- 超阈值 → 压缩一次（每次问答至多一次低温度短输出调用）
"""

from core.config import settings
from core.llm_client import llm_client

_COMPRESS_PROMPT = """你是对话摘要器。将【已有摘要】与【新对话段】合并为一份摘要：
1. 保留所有关键实体、数据、结论（如数字、街道名、系统名、理论名）
2. 删除寒暄与重复内容
3. 只输出摘要正文，不超过300字"""

_MAX_SUMMARY_CHARS = 600


def _format_turns(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {str(m.get('content', ''))[:200]}")
    return "\n".join(lines)


def prepare(
    history: list[dict] | None,
    existing_summary: str | None = None,
) -> tuple[list[dict], str]:
    """
    记忆准备入口：窗口裁剪 + 必要时滚动压缩

    Args:
        history: 全量历史消息 [{"role","content"},...]
        existing_summary: 上一轮的滚动摘要

    Returns:
        (windowed_history, new_summary)
    """
    if not history:
        return [], existing_summary or ""

    cfg = settings.memory_config
    threshold = int(cfg.get("compress_threshold", 14))
    keep = int(cfg.get("recent_turns", 6)) * 2  # 1轮 = user+assistant 两条

    if len(history) <= threshold:
        return history, existing_summary or ""

    to_compress = history[:-keep]
    window = list(history[-keep:])

    new_summary = (existing_summary or "").strip()
    try:
        user_content = (
            f"【已有摘要】\n{new_summary or '（无）'}\n\n"
            f"【新对话段】\n{_format_turns(to_compress)}"
        )
        part = llm_client.chat(
            [
                {"role": "system", "content": _COMPRESS_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        if part and part.strip():
            new_summary = part.strip()
    except Exception:
        # LLM 失败时兜底：截断式伪摘要，保证管线不中断
        fallback = _format_turns(to_compress)
        new_summary = (new_summary + "\n" + fallback).strip()

    if len(new_summary) > _MAX_SUMMARY_CHARS:
        new_summary = new_summary[:_MAX_SUMMARY_CHARS]
    return window, new_summary
