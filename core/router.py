"""
查询路由器（P1）
用一次低成本 LLM 调用对用户问题分类，决定检索范围；追问时改写为语义完整的问题。

类型定义：
- report:     一稿/选题报告的框架、观点、结论、方法类问题
- case:       案例事实细节类问题（数据、时间线、主体、做法）
- literature: 理论/概念的学术出处与文献内容类问题
- followup:   含指代词的追问（"那它的作用呢？"）

快速通道：超短问题（<=6字）且不含指代词 → 跳过 LLM 调用，直接全库检索。
（中文问题普遍短小，阈值过宽会让路由器形同虚设，故仅放行"报表通是什么"级别的查询。）
JSON 解析失败 / 类型非法 → 兜底全库检索（type=None），保证可用性优先。
"""

import json
import re

from core.llm_client import llm_client

# 出现即视为可能的追问（需结合上下文）
PRONOUNS = ("他", "她", "它", "这", "那", "其", "该")

# 分类是 max_tokens=200 的廉价调用，重试次数比常规对话更宽，减少 429 导致的静默兜底
_ROUTE_RETRIES = 5

_ROUTE_PROMPT = """你是查询分类器。只输出 JSON，不要输出任何其他内容：
{"type": "report|case|literature|followup", "rewritten_query": "改写后的语义完整的独立问题"}

分类标准：
- report: 询问分析框架、核心结论、观点、研究方法、实践审思、研究意义等团队报告内容。
  注意：只要问的是报告的组成部分（框架/方法/意义/结论/审思），即使句中出现"案例"字样也判 report
- case: 询问案例本身的事实细节（数据、时间、街道社区、系统功能、做法成效）
- literature: 询问某个理论或概念的学术出处、某篇文献的内容与作者
- followup: 问题含指代词（它/这个/该/其等）且指代上文内容，或离开对话历史无法理解。
  注意：只要含指代上游内容的指代词就判 followup，不要按问题主题改判其他类型

改写规则：若问题含指代词，结合对话历史把问题改写为不依赖上下文的完整问法；否则原样返回。"""


def _extract_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no json found")
    return json.loads(match.group())


def classify_query(query: str, history: list | None = None) -> dict:
    """
    对用户问题分类并按需改写

    Returns:
        {"type": "report/case/literature/followup/None",
         "rewritten_query": str,
         "fast_path": bool}
        type=None 表示全库检索（快速通道或兜底）
    """
    q = query.strip()
    # 快速通道：超短且无指代词 → 无需 LLM 判断
    if len(q) <= 6 and not any(p in q for p in PRONOUNS):
        return {"type": None, "rewritten_query": q, "fast_path": True}

    messages = [{"role": "system", "content": _ROUTE_PROMPT}]
    if history:
        recent = history[-4:]
        ctx = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:120]}" for m in recent
        )
        messages.append(
            {"role": "user", "content": f"对话历史：\n{ctx}\n\n当前问题：{q}"}
        )
    else:
        messages.append({"role": "user", "content": f"当前问题：{q}"})

    try:
        resp = llm_client.chat(
            messages, temperature=0.0, max_tokens=200, retries=_ROUTE_RETRIES
        )
        data = _extract_json(resp)
        rtype = data.get("type")
        if rtype not in ("report", "case", "literature", "followup"):
            rtype = None
        rewritten = str(data.get("rewritten_query") or q).strip() or q
        return {"type": rtype, "rewritten_query": rewritten, "fast_path": False}
    except Exception:
        # 兜底：分类失败不影响问答，退回全库检索
        return {"type": None, "rewritten_query": q, "fast_path": False}
