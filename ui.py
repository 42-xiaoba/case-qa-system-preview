"""
Streamlit 前端界面
提供智能案例问答系统的交互界面。
支持两种运行模式：
  - API 模式：依赖 FastAPI 后端（本地开发）
  - 直连模式：直接调用 LLM API（Streamlit Cloud 部署）
"""

import base64
import codecs
import json
import queue
import re
import threading
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components

# ==================== 页面配置（必须是第一个 Streamlit 命令） ====================

st.set_page_config(
    page_title="智能案例问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 密钥注入 ====================
# 从 st.secrets 读取并注入环境变量（Streamlit Cloud 部署用）
# 用 try-except 逐个读取，避免某个 key 缺失导致全部注入失败
import os as _os
_secrets_errors = []
for _key in ("GLM_API_KEY", "GLM_V_API_KEY", "OEPNROUTER_API_KEY"):
    try:
        _val = st.secrets[_key]
        if _val:
            _os.environ[_key] = str(_val).strip()
    except Exception as e:
        _secrets_errors.append(f"{_key}: {e}")

from core.llm_client import (
    llm_client,
    get_vision_llm_client,
    ModelFallbackSignal,
    MODEL_FALLBACK_SIGNAL,
    API_FALLBACK_MARKER,
)
from core.memory import prepare as memory_prepare
from core.pipeline import (
    FOLLOWUP_TAG,
    append_followup_instruction,
    build_answer_messages_routed,
)
from core.prompt_manager import prompt_manager

# ==================== CSS ====================
# 策略：
# 1. 页面级 overflow:hidden 禁止整页滚动
# 2. 列容器设固定高度边界
# 3. st.container(height=700) 会在内联样式写 overflow:auto，创建可滚动容器
# 4. CSS !important 覆盖 700px 为视口高度（CSS !important 可覆盖内联样式）

CUSTOM_CSS = """
<style>
    /* ========== 侧边栏悬停弹出 ========== */
    section[data-testid="stSidebar"] {
        width: 60px !important;
        min-width: 60px !important;
        max-width: 60px !important;
        transition: width 0.3s ease, min-width 0.3s ease, max-width 0.3s ease !important;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"]:hover {
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        overflow: auto !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding: 0.5rem !important;
        position: relative !important;
        height: 100vh !important;
    }
    /* 收起时隐藏侧边栏内所有内容（除 hamburger 和竖排提示） */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] .stAlert,
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] .stImage,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] [data-testid="stRadio"],
    section[data-testid="stSidebar"] .stCaption {
        opacity: 0;
        transition: opacity 0.25s ease 0.05s;
        pointer-events: none;
    }
    /* 展开时恢复显示 */
    section[data-testid="stSidebar"]:hover .stMarkdown,
    section[data-testid="stSidebar"]:hover .stButton,
    section[data-testid="stSidebar"]:hover .stAlert,
    section[data-testid="stSidebar"]:hover .stFileUploader,
    section[data-testid="stSidebar"]:hover .stImage,
    section[data-testid="stSidebar"]:hover [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"]:hover [data-testid="stFileUploaderDropzoneInstructions"],
    section[data-testid="stSidebar"]:hover [data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"]:hover [data-testid="stRadio"],
    section[data-testid="stSidebar"]:hover .stCaption {
        opacity: 1;
        pointer-events: auto;
    }
    .sidebar-hamburger {
        font-size: 24px;
        text-align: center;
        padding: 8px 0;
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    /* 收起时的竖排提示文字（用伪元素，避免被 .stMarkdown 的 opacity:0 影响） */
    section[data-testid="stSidebar"]::after {
        content: "图片上传，切换pdf等功能在此处";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        writing-mode: vertical-rl;
        text-orientation: upright;
        font-size: 0.75rem;
        color: #999;
        white-space: nowrap;
        letter-spacing: 0.2em;
        opacity: 1;
        transition: opacity 0.2s ease;
        pointer-events: none;
        z-index: 10;
    }
    section[data-testid="stSidebar"]:hover::after {
        opacity: 0;
    }

    /* 文档选择：当前展示的文档选项高亮标注 */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(88, 166, 255, 0.18);
        border-radius: 6px;
        box-shadow: inset 0 0 0 1px rgba(88, 166, 255, 0.45);
    }

    /* ========== 整页固定：禁止页面级滚动 ========== */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
        margin: 0 !important;
    }
    .stApp {
        height: 100vh !important;
        overflow: hidden !important;
    }
    .stApp > header {
        display: none !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0.5rem 1rem !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    /* ========== 列容器：固定高度边界 ========== */
    div[data-testid="stHorizontalBlock"] {
        height: calc(100vh - 100px) !important;
        overflow: hidden !important;
    }
    div[data-testid="column"] {
        height: 100% !important;
        overflow: hidden !important;
    }

    /* ========== 覆盖 st.container(height=700) 为视口高度 ========== */
    /* st.container(height=700) 生成 <div style="...overflow:auto; height:700px;"> */
    /* CSS !important 可覆盖内联样式 */
    div[data-testid="column"]:nth-child(1) div[style*="overflow"] {
        height: calc(100vh - 130px) !important;
        max-height: calc(100vh - 130px) !important;
        overflow-y: auto !important;
    }
    /* 左栏滚动条暗色 */
    div[data-testid="column"]:nth-child(1) div[style*="overflow"]::-webkit-scrollbar {
        width: 6px;
    }
    div[data-testid="column"]:nth-child(1) div[style*="overflow"]::-webkit-scrollbar-track {
        background: #161b22;
    }
    div[data-testid="column"]:nth-child(1) div[style*="overflow"]::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 3px;
    }

    /* ========== 覆盖 st.pdf(height=700) iframe 为视口高度 ========== */
    div[data-testid="column"]:nth-child(2) iframe {
        height: calc(100vh - 60px) !important;
        max-height: calc(100vh - 60px) !important;
        border: none !important;
    }
    /* 如果 st.pdf 也生成了带 overflow 的容器，一并覆盖 */
    div[data-testid="column"]:nth-child(2) div[style*="overflow"] {
        height: calc(100vh - 60px) !important;
        max-height: calc(100vh - 60px) !important;
        overflow-y: auto !important;
    }

    /* ========== PDF 缩放按钮（叠加在 st.pdf 之上） ========== */
    .pdf-zoom-controls {
        position: absolute;
        top: 12px;
        left: 12px;
        display: flex;
        gap: 6px;
        z-index: 100;
    }
    .pdf-zoom-btn {
        width: 32px;
        height: 32px;
        border-radius: 6px;
        background: rgba(40, 40, 40, 0.9);
        color: #fff;
        border: 1px solid #555;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s;
        user-select: none;
    }
    .pdf-zoom-btn:hover {
        background: rgba(70, 70, 70, 0.95);
    }
    .pdf-zoom-btn:active {
        background: rgba(100, 100, 100, 0.95);
    }
    .pdf-zoom-label {
        color: #ccc;
        font-size: 0.75rem;
        align-self: center;
        background: rgba(40, 40, 40, 0.9);
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #555;
        min-width: 50px;
        text-align: center;
    }
    .pdf-hint {
        position: absolute;
        bottom: 12px;
        right: 12px;
        background: rgba(0,0,0,0.7);
        color: #ccc;
        font-size: 0.7rem;
        padding: 4px 8px;
        border-radius: 4px;
        pointer-events: none;
        z-index: 100;
    }

    /* ========== 禁用 Streamlit 运行时的全屏变暗遮罩 ========== */
    /* 1. 隐藏右上角运行状态指示器 */
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    /* 2. 移除主容器在 running 状态下的半透明遮罩 */
    [data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
        filter: none !important;
    }
    /* 3. 移除 running 时盖在内容上的灰色 overlay（streamlit 1.59 用 .stSpinner 容器） */
    .stSpinner,
    div[data-testid="stSpinnerContainer"] {
        background: transparent !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }
    /* 4. 阻止 iframe / block 在运行时被加暗色蒙层 */
    iframe,
    [data-testid="stHorizontalBlock"],
    [data-testid="stVerticalBlock"],
    section[data-testid="stMain"],
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        filter: none !important;
    }
    /* 5. 兜底：所有 running 状态下出现的固定定位遮罩层一律透明 */
    div[aria-live="polite"],
    div[data-testid="stToastContainer"] {
        background: transparent !important;
    }
</style>
"""


def load_custom_css():
    """加载自定义 CSS"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== 后端 API 配置 ====================

API_BASE_URL = "http://localhost:8000"

# 右侧 PDF 预览的文档映射：选项名 → 根目录文件名
DOC_FILES = {
    "选题报告": "case_report.pdf",
    "案例报告（一稿）": "清华案例分析报告一稿.pdf",
}


def check_api_health() -> bool:
    """检查后端服务是否正常运行"""
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE_URL}/api/health", timeout=5)
            return resp.json().get("status") == "ok"
    except Exception:
        return False


_WAITING_BASE = "⏳ 正在思考你的问题，可能会有些慢，请不要着急"
_WAITING_BUSY_SUFFIX = "（当前访问量过大，请耐心等待）"
_WAITING_MSG = _WAITING_BASE + "...\n\n"
# 主模型首次请求失败（限流/超时触发模型降级）后切换为此提示，末尾三点循环动画不变
_WAITING_MSG_BUSY = _WAITING_BASE + _WAITING_BUSY_SUFFIX + "...\n\n"

# 初始问候语 + 功能介绍
GREETING_MESSAGE = """你好！我是智能案例问答助手，请随时向我提问关于案例的问题。

---

### 📖 功能介绍

**1. 智能案例问答（文本）**
在下方输入框输入你的问题，例如"这个案例的研究意义是什么？""交易成本理论如何应用？"，我会基于案例文本给出专业、有深度的回答。

**2. 图片识别问答（视觉）**
点击左侧栏「🖼️ 添加图片」上传图片（支持 PNG/JPG/WebP/GIF，每次最多1张），然后输入问题即可让我识别图片内容并回答。适合上传案例中的图表、流程图、截图等视觉内容提问。

**3. PDF 文档预览**
右侧栏展示文档 PDF，可在左侧栏「📄 选择文档」中切换「选题报告」/「案例报告（一稿）」，当前展示的文档会高亮标注。支持以下操作：
- 滚动鼠标滚轮：上下翻阅 PDF 内容
- 点击「➕」/「➖」按钮：放大或缩小 PDF
- 按住 Shift + 滚动鼠标滚轮：横向滚动放大后的 PDF

**4. 对话管理**
- 左侧栏「🗑️ 清空对话」：清空当前所有对话记录，重新开始

---

💡 **使用提示**：问题越具体，回答越精准。涉及案例中的数据、人物、政策时，建议直接引用相关关键词提问。"""


def send_chat_request_stream_direct(
    query: str,
    history: list | None = None,
    notice: dict | None = None,
    existing_summary: str | None = None,
    summary_out: dict | None = None,
):
    """直连模式流式生成器（记忆压缩 → 路由 → 检索 → 预算制组装）

    notice: 降级通知字典。发生模型降级时置 notice["busy"]=True，
            前端动画循环据此把等待提示切换为"访问量过大"版本。
    existing_summary/summary_out: 记忆摘要的传入与传出容器。生成器会在
            后台线程中执行，不能直接读写 st.session_state，因此由调用方
            传入当前摘要，并用 summary_out["value"] 带回压缩后的新摘要，
            由主线程在回答完成时统一写回会话状态。
    """
    yield _WAITING_MSG
    windowed, new_summary = memory_prepare(history or [], existing_summary)
    if summary_out is not None:
        summary_out["value"] = new_summary
    messages, _docs, _route = build_answer_messages_routed(
        query,
        history=windowed,
        history_summary=new_summary or None,
    )
    messages = append_followup_instruction(messages)
    try:
        for chunk in llm_client.chat_stream(messages):
            if isinstance(chunk, ModelFallbackSignal):
                if notice is not None:
                    notice["busy"] = True
                continue
            yield chunk
    except Exception as e:
        yield f"\n\n[错误] 模型调用失败: {e}"


def send_chat_request_stream_api(query: str, history: list | None = None, notice: dict | None = None):
    """API 模式流式生成器

    notice: 降级通知字典。检测到后端发来的降级标记时置 notice["busy"]=True。
    """
    yield _WAITING_MSG
    payload = {"query": query, "history": history or []}
    marker = API_FALLBACK_MARKER
    keep = len(marker) - 1  # 缓存尾部字符，防止标记被字节块边界截断
    decoder = codecs.getincrementaldecoder("utf-8")()
    buf = ""
    try:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/api/chat/stream",
                json=payload,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_bytes():
                    buf += decoder.decode(raw)
                    while True:
                        idx = buf.find(marker)
                        if idx >= 0:
                            head, buf = buf[:idx], buf[idx + len(marker):]
                            if head:
                                yield head
                            if notice is not None:
                                notice["busy"] = True
                            continue
                        if len(buf) > keep:
                            yield buf[:-keep]
                            buf = buf[-keep:]
                        break
                try:
                    buf += decoder.decode(b"", final=True)
                except Exception:
                    pass
                if buf:
                    yield buf
    except httpx.HTTPStatusError as e:
        yield f"\n\n[错误] 请求失败 (HTTP {e.response.status_code})"
    except httpx.ConnectError:
        yield "\n\n[错误] 无法连接到后端服务，请确保 FastAPI 已启动"
    except Exception as e:
        yield f"\n\n[错误] {str(e)}"


def send_vision_chat_stream_direct(query: str, image_data_url: str, history: list | None = None):
    """直连模式视觉流式生成器"""
    yield _WAITING_MSG
    messages = prompt_manager.build_vision_messages(
        user_query=query, image_data_url=image_data_url, history=history or []
    )
    messages = append_followup_instruction(messages)
    try:
        for chunk in get_vision_llm_client().chat_stream(messages):
            yield chunk
    except Exception as e:
        yield f"\n\n[错误] 视觉模型调用失败: {e}"


def send_vision_chat_stream_api(query: str, image_data_url: str, history: list | None = None):
    """API 模式视觉流式生成器"""
    yield _WAITING_MSG
    payload = {"query": query, "image": image_data_url, "history": history or []}
    try:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/api/chat/vision/stream",
                json=payload,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8")
    except httpx.HTTPStatusError as e:
        yield f"\n\n[错误] 请求失败 (HTTP {e.response.status_code})"
    except httpx.ConnectError:
        yield "\n\n[错误] 无法连接到后端服务，请确保 FastAPI 已启动"
    except Exception as e:
        yield f"\n\n[错误] {str(e)}"


# ==================== 回答任务：跨 rerun 的流式消费 ====================
# Streamlit 中任何组件交互（切换PDF、缩放、上传图片等）都会触发脚本整体重跑，
# 正在进行的回答若依赖单次脚本运行内的循环就会被中断。这里把"消费流"全部
# 移交后台线程，并把 队列/线程/已累计文本 存入 session_state：
# 重跑后检测到未完成任务即重建气泡继续渲染，消费完毕再写入消息历史。

# 首个内容块到达后、流式输出期间显示的固定前缀
_THINKING_PREFIX = "✅ 已完成思考\n\n"
# 全部内容接收完毕但正文为空时的兜底文案
_EMPTY_REPLY = "（未获取到回复）"


def _start_answer_task(stream, notice=None, summary_out=None, question=None):
    """启动后台线程完整消费流式生成器，返回存入 session_state 的任务字典"""
    task = {
        "queue": queue.Queue(),
        "thread": None,
        "text": "",  # 已累计的回答正文（仅主线程在渲染循环中更新）
        "finished": False,
        "notice": notice if notice is not None else {"busy": False},
        "summary_out": summary_out if summary_out is not None else {},
        "question": question or "",  # 触发本轮回答的用户问题（用于生成延伸问题）
        # 延伸问题预生成：回答过半即后台提前生成，正式回答完成时通常已就绪，
        # 卡片随回答完成即时出现；未就绪时 finalize 再短暂等待
        "fu_thread": None,
        "fu_result": None,
    }

    def _fu_worker(partial: str):
        task["fu_result"] = _generate_followups(task["question"], partial)

    def consume():
        try:
            next(stream)  # 丢弃 _WAITING_MSG 占位片段
            text = ""
            fu_started = False
            while True:
                try:
                    chunk = next(stream)
                except StopIteration:
                    break
                text += chunk
                if (
                    not fu_started
                    and task["question"]
                    and len(text) >= 400
                ):
                    fu_started = True
                    task["fu_thread"] = threading.Thread(
                        target=_fu_worker, args=(text,), daemon=True
                    )
                    task["fu_thread"].start()
                task["queue"].put(("chunk", chunk))
            task["queue"].put(("done", text))
        except Exception as e:  # noqa: BLE001
            task["queue"].put(("error", str(e)))

    task["thread"] = threading.Thread(target=consume, daemon=True)
    task["thread"].start()
    return task


def _drain_answer_task(task, placeholder):
    """渲染回答进度直至完成：无内容时播放三点动画，收到内容后增量刷新。

    本循环被组件交互触发的重跑打断时直接随脚本退出，不做任何清理——
    任务状态在 session_state 中完好，下次运行由恢复分支续接。"""
    if task["finished"]:
        placeholder.write(_THINKING_PREFIX + _hide_followup_tail(task["text"]))
        return
    dots = 0
    while True:
        if task["text"]:
            placeholder.write(_THINKING_PREFIX + _hide_followup_tail(task["text"]))
        else:
            dots = (dots % 3) + 1
            base_msg = _WAITING_BASE + _WAITING_BUSY_SUFFIX if task["notice"]["busy"] else _WAITING_BASE
            placeholder.write(base_msg + "." * dots)
        try:
            kind, payload = task["queue"].get(timeout=1.0)
        except queue.Empty:
            if not task["thread"].is_alive() and task["queue"].empty():
                # 线程意外终止且没有产出完成标记的兜底，避免动画永远转圈
                task["text"] = task["text"] or _EMPTY_REPLY
                task["finished"] = True
                placeholder.write(_THINKING_PREFIX + _hide_followup_tail(task["text"]))
                return
            continue
        if kind == "chunk":
            task["text"] += payload
            placeholder.write(_THINKING_PREFIX + _hide_followup_tail(task["text"]))
        elif kind == "done":
            task["text"] = payload or task["text"] or _EMPTY_REPLY
            task["finished"] = True
            placeholder.write(_THINKING_PREFIX + _hide_followup_tail(task["text"]))
            return
        else:  # error
            task["text"] = f"[错误] {payload}"
            task["finished"] = True
            placeholder.write(_THINKING_PREFIX + task["text"])
            return


# ==================== 延伸问题（追问卡片） ====================

# 追问建议进程内缓存：key=问题+回答摘要 → {"related","new"}，避免同一问答重复生成浪费配额。
# 用模块级字典而非 session_state：预生成工作在后台线程中执行，不能触碰 st.session_state
_FOLLOWUP_CACHE_MAX = 40
_FOLLOWUP_CACHE: dict = {}


def _hide_followup_tail(text: str) -> str:
    """流式显示时隐藏延伸问题块：截断完整标记及其后内容，
    并处理被 chunk 切断的标记前缀，避免半截标签闪现"""
    idx = text.find(FOLLOWUP_TAG)
    if idx != -1:
        return text[:idx]
    for i in range(min(len(FOLLOWUP_TAG) - 1, len(text)), 0, -1):
        if text.endswith(FOLLOWUP_TAG[:i]):
            return text[:-i]
    return text


def _parse_followup_tail(text: str):
    """从全文尾部提取延伸问题块 → (剥离后的正文, followups|None)；
    无标记或解析失败返回原文与 None（调用方走独立生成兜底）"""
    idx = text.find(FOLLOWUP_TAG)
    if idx == -1:
        return text, None
    body = text[:idx].rstrip()
    tail = text[idx + len(FOLLOWUP_TAG):].split("</followups>")[0]
    m = re.search(r"\{.*\}", tail, re.S)
    data = json.loads(m.group(0)) if m else {}
    result = {
        "related": str(data.get("related", "")).strip()[:60],
        "new": str(data.get("new", "")).strip()[:60],
    }
    if not result["related"] and not result["new"]:
        return text, None
    return (body if body.strip() else text), result


def _generate_followups(question: str, answer: str):
    """基于一轮问答生成两个延伸问题：相关话题深入（related）+ 新话题拓展（new）。

    关闭思考模式以缩短延迟；限流/解析失败等异常一律返回 None，
    前端静默跳过卡片，不影响正常回答展示。"""
    cache = _FOLLOWUP_CACHE
    key = f"{question[:100]}|{answer[:200]}"
    if key in cache:
        return cache[key]
    prompt = (
        "请根据下面这轮问答，生成两个适合用户继续提问的中文问题：\n"
        '1. "related"：紧扣本次回答内容的延伸追问，帮助用户深入了解细节；\n'
        '2. "new"：切换到一个与本次话题相关但角度全新的新话题问题，帮助用户拓展视野。\n'
        '要求：每个问题不超过25个字，独立成句、不使用指代词；只输出 JSON，'
        '格式为 {"related":"问题一","new":"问题二"}，不要输出任何解释。\n\n'
        f"【用户问题】{question}\n【助手回答】{answer[:800]}"
    )
    try:
        raw = llm_client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            retries=1,
            # 附加智谱思考控制块：config 中 model.thinking=false 时该块即为
            # "显式禁用"，防止服务端默认开启思考吃掉 max_tokens 导致正文为空
            use_thinking=True,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        result = {
            "related": str(data.get("related", "")).strip()[:60],
            "new": str(data.get("new", "")).strip()[:60],
        }
        if not result["related"] and not result["new"]:
            print(f"[followups] 模型输出无法解析出问题: {raw[-120:]!r}", flush=True)
            return None
        if len(cache) >= _FOLLOWUP_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result
    except Exception as e:
        print(f"[followups] 生成失败: {type(e).__name__}: {str(e)[:120]}")
        return None


def render_followup_cards(followups: dict):
    """在回答下方渲染两张可点击的延伸问题小卡片。

    卡片运行在 components.html 的 iframe 中：点击时通过 window.parent 定位
    页面底部的聊天输入框 textarea，用原生 value setter 写入并派发 input 事件
    （React 受控组件的标准注入方式），实现"点卡片 → 问题填入输入框"。"""
    cards = []
    if followups.get("related"):
        cards.append(("🔗 相关延伸", followups["related"]))
    if followups.get("new"):
        cards.append(("🌱 新话题", followups["new"]))
    if not cards:
        return

    data_json = json.dumps([[tag, q] for tag, q in cards], ensure_ascii=False)
    height = 52 * len(cards) + 16
    html = f"""
<div id="fu-wrap"></div>
<style>
    #fu-wrap {{ display: flex; flex-direction: column; gap: 8px; padding: 6px 0 2px; }}
    .fu-card {{
        display: flex; align-items: center; gap: 8px;
        border: 1px solid rgba(90, 140, 220, 0.35);
        border-radius: 10px;
        background: rgba(90, 140, 220, 0.08);
        padding: 8px 12px;
        cursor: pointer;
        user-select: none;
        transition: background 0.15s ease, transform 0.1s ease, border-color 0.15s ease;
    }}
    .fu-card:hover {{
        background: rgba(90, 140, 220, 0.18);
        border-color: rgba(90, 140, 220, 0.65);
        transform: translateY(-1px);
    }}
    .fu-card:active {{ transform: translateY(0); opacity: 0.85; }}
    .fu-tag {{
        flex-shrink: 0;
        font-size: 0.72rem;
        line-height: 1.4;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(90, 140, 220, 0.22);
        white-space: nowrap;
    }}
    .fu-text {{
        font-size: 0.85rem;
        line-height: 1.45;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
</style>
<script>
(function () {{
    var data = {data_json};
    var wrap = document.getElementById("fu-wrap");
    data.forEach(function (pair) {{
        var card = document.createElement("div");
        card.className = "fu-card";
        var tag = document.createElement("span");
        tag.className = "fu-tag";
        tag.textContent = pair[0];
        var text = document.createElement("span");
        text.className = "fu-text";
        text.textContent = pair[1];
        text.title = pair[1];
        card.appendChild(tag);
        card.appendChild(text);
        card.addEventListener("click", function () {{ fill(pair[1]); }});
        wrap.appendChild(card);
    }});

    function fill(q) {{
        try {{
            var p = window.parent;
            if (!p || !p.document) return;
            // 定位底部聊天输入框：优先官方 testid，失败则取视口最下方的 textarea
            var ta = p.document.querySelector(
                '[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"]');
            if (!ta) {{
                var best = null, top = -1;
                p.document.querySelectorAll("textarea").forEach(function (t) {{
                    var r = t.getBoundingClientRect();
                    if (r.top > top) {{ top = r.top; best = t; }}
                }});
                ta = best;
            }}
            if (!ta) return;
            ta.focus();
            var setter = Object.getOwnPropertyDescriptor(
                p.HTMLTextAreaElement.prototype, "value").set;
            setter.call(ta, q);
            ta.dispatchEvent(new p.Event("input", {{ bubbles: true }}));
        }} catch (e) {{ /* 静默失败：不影响回答区 */ }}
    }}
}})();
</script>
"""
    components.html(html, height=height, scrolling=False)


def _finalize_answer_task():
    """把完成的回答写入消息历史并清理任务（含记忆摘要写回会话状态）"""
    task = st.session_state.get("answer_task")
    if task is None:
        return
    new_summary = task.get("summary_out", {}).get("value")
    if new_summary:
        st.session_state["history_summary"] = new_summary
    text = task["text"] or _EMPTY_REPLY
    # 路径一（零成本）：回答末尾若带有随答延伸块，直接解析剥离（历史正文不含标记）
    followups = None
    if text != _EMPTY_REPLY and not text.startswith("[错误]"):
        clean, parsed = _parse_followup_tail(text)
        if parsed:
            followups = parsed
            text = clean
    message = {"role": "assistant", "content": text}
    # 路径二（零等待）：预生成线程在回答期间已启动，此刻通常已完成；
    # 极少数未完成时最多等 45s（与旧同步兜底耗时同量级），超时放弃本轮卡片
    if followups is None and task.get("fu_thread") is not None:
        task["fu_thread"].join(timeout=45)
        if task.get("fu_result") is not None:
            followups = task["fu_result"]
    # 路径三（同步兜底）：回答过短未触发预生成时，才现场补一次独立生成；
    # 预生成已启动但失败的轮次不再重试，避免限流下双倍等待
    if (
        followups is None
        and text != _EMPTY_REPLY
        and not text.startswith("[错误]")
        and task.get("question")
        and task.get("fu_thread") is None
    ):
        followups = _generate_followups(task["question"], text)
    if followups:
        message["followups"] = followups
    st.session_state.messages.append(message)
    st.session_state.pop("answer_task", None)
    # 消息循环在本函数之前已执行完毕，新增的延伸问题卡片必须重跑一次才会渲染；
    # 此时任务已清理，重跑不会再次进入本函数，无死循环风险
    st.rerun()


# ==================== 会话状态初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": GREETING_MESSAGE}
    ]

if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False


def render_message_content(content):
    """渲染消息内容：支持纯文本和多模态（文本+图片）"""
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                st.write(part["text"])
            elif part.get("type") == "image_url":
                st.image(part["image_url"]["url"])
    else:
        st.write(content)


# ==================== 侧边栏 ====================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-hamburger">☰</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## 💬 智能案例问答系统")
    st.markdown("---")

    # ---- 图片上传（最多1张） ----
    st.markdown("### 🖼️ 添加图片")
    _vision_client = get_vision_llm_client()
    if _vision_client is not None:
        # 用计数器作为 uploader key 的一部分，点击"移除"时递增计数器即可强制重置 widget
        if "image_uploader_counter" not in st.session_state:
            st.session_state["image_uploader_counter"] = 0
        uploader_key = f"vision_image_uploader_{st.session_state['image_uploader_counter']}"

        uploaded_image = st.file_uploader(
            "选择图片（最多1张）",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key=uploader_key,
        )
        if uploaded_image is not None:
            img_bytes = uploaded_image.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime_type = uploaded_image.type or "image/png"
            st.session_state["pending_image"] = f"data:{mime_type};base64,{img_b64}"
            st.image(uploaded_image, caption="已选择", use_container_width=True)
            if st.button("🗑️ 移除图片", use_container_width=True):
                # 清除待发送图片 + 递增计数器让 uploader 在下一次 rerun 时生成新 key，从而被重置为空
                st.session_state.pop("pending_image", None)
                st.session_state["image_uploader_counter"] += 1
                st.rerun()
        else:
            st.session_state.pop("pending_image", None)
    else:
        st.warning("⚠️ 视觉功能未启用")
        # 诊断信息：帮助定位密钥读取失败的原因
        _v_key_env = bool(_os.environ.get("GLM_V_API_KEY"))
        _v_key_secret = False
        try:
            _v_key_secret = bool(st.secrets.get("GLM_V_API_KEY"))
        except Exception:
            pass
        st.caption(f"环境变量 GLM_V_API_KEY: {'✅ 已读取' if _v_key_env else '❌ 未读取'}")
        st.caption(f"st.secrets GLM_V_API_KEY: {'✅ 已读取' if _v_key_secret else '❌ 未读取'}")
        if _secrets_errors:
            st.caption(f"注入异常: {'; '.join(_secrets_errors)}")
        st.caption("请在 Secrets 中配置 GLM_V_API_KEY")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": GREETING_MESSAGE}
        ]
        st.session_state.pop("pending_image", None)
        # 同步丢弃进行中的回答任务：后台线程会自然结束，结果不再写入历史
        st.session_state.pop("answer_task", None)
        st.session_state["image_uploader_counter"] += 1
        st.rerun()
    st.markdown("---")

    # ---- 后端健康检查（静默执行，仅用于选择 API/直连模式） ----
    st.session_state.api_healthy = check_api_health()

    # ---- 选择文档：切换右侧 PDF 预览，选中项高亮标注 ----
    st.markdown("### 📄 选择文档")
    active_doc = st.radio(
        "选择右侧展示的文档",
        options=list(DOC_FILES.keys()),
        key="active_doc",
        label_visibility="collapsed",
    )
    st.caption(f"当前展示：{DOC_FILES[active_doc]}")
    st.markdown("---")

    st.markdown(
        '<div style="font-size: 0.8rem; color: #999; text-align: center;">'
        "智能案例问答系统 v0.3.1<br>团队成员：<br>卜天伊 冯思杰 等</div>",
        unsafe_allow_html=True,
    )


# ==================== 主界面 ====================

load_custom_css()

left_col, right_col = st.columns([0.6, 0.4], gap="medium", vertical_alignment="top")


# ==================== 左栏：AI 问答 ====================

with left_col:
    st.caption("💬 案例问答 · SenseNova 6.8 FlashLite")

    # 可滚动聊天容器，固定像素高度
    chat_container = st.container(height=750)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                render_message_content(msg["content"])
                if msg.get("followups"):
                    render_followup_cards(msg["followups"])

    # ---- 恢复未完成的回答：任何组件交互（切换PDF/缩放/上传等）触发重跑后，
    # 从这里凭 session_state 中的任务续接渲染，直到消费完毕再写入历史 ----
    pending_task = st.session_state.get("answer_task")
    if pending_task is not None:
        with chat_container:
            with st.chat_message("assistant"):
                resume_placeholder = st.empty()
        _drain_answer_task(pending_task, resume_placeholder)
        _finalize_answer_task()

    if prompt := st.chat_input("请输入你的问题，例如：这个案例的研究意义是什么？"):
        pending_image = st.session_state.pop("pending_image", None)

        if pending_image:
            # 多模态消息：文本 + 图片
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": pending_image}},
            ]
        else:
            user_content = prompt

        st.session_state.messages.append({"role": "user", "content": user_content})

        with chat_container:
            with st.chat_message("user"):
                render_message_content(user_content)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["content"] != GREETING_MESSAGE  # 问候语不作为上下文发送
        ]
        fallback_notice = {"busy": False}
        summary_out = {}

        # 选择对应的流式生成器（视觉/文本 × API/直连）
        if pending_image and get_vision_llm_client() is not None:
            # 有图片：走视觉模型
            if st.session_state.api_healthy:
                stream = send_vision_chat_stream_api(prompt, pending_image, history)
            else:
                stream = send_vision_chat_stream_direct(prompt, pending_image, history)
        else:
            # 无图片：走文本模型（主模型受限/超时时自动降级到备用模型）
            if st.session_state.api_healthy:
                stream = send_chat_request_stream_api(prompt, history, notice=fallback_notice)
            else:
                stream = send_chat_request_stream_direct(
                    prompt,
                    history,
                    notice=fallback_notice,
                    existing_summary=st.session_state.get("history_summary"),
                    summary_out=summary_out,
                )

        # 流的消费全部移交后台线程并存入 session_state：
        # 即使期间用户切换PDF/缩放等触发重跑，回答也会在下一轮运行中续接完成
        task = _start_answer_task(stream, notice=fallback_notice,
                                  summary_out=summary_out, question=prompt)
        st.session_state["answer_task"] = task

        with chat_container:
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
        _drain_answer_task(task, answer_placeholder)
        _finalize_answer_task()


# ==================== 右栏：PDF 预览（缩放按钮 + Shift 横向滚动） ====================

with right_col:
    # 根据侧栏「选择文档」的选中项切换 PDF（选中项在侧栏高亮标注）
    active_doc = st.session_state.get("active_doc", "选题报告")
    pdf_path = Path(__file__).resolve().parent / DOC_FILES.get(active_doc, "case_report.pdf")

    if not pdf_path.exists():
        st.error(f"❌ 未找到 `{pdf_path.name}`，请放入项目根目录")
    else:
        # 在 PDF 上方放缩放控制条（用 columns 让按钮和 PDF 在同一列）
        zoom_col1, zoom_col2, zoom_col3 = st.columns([1, 1, 6])
        with zoom_col1:
            zoom_in_btn = st.button("➕", key="pdf_zoom_in", help="放大 PDF")
        with zoom_col2:
            zoom_out_btn = st.button("➖", key="pdf_zoom_out", help="缩小 PDF")
        with zoom_col3:
            st.caption("滚轮上下滚动 · Shift+滚轮横向滚动")

        # 初始化缩放状态
        if "pdf_zoom" not in st.session_state:
            st.session_state["pdf_zoom"] = 1.0

        # 处理缩放按钮
        if zoom_in_btn:
            st.session_state["pdf_zoom"] = min(3.0, st.session_state["pdf_zoom"] + 0.2)
        if zoom_out_btn:
            st.session_state["pdf_zoom"] = max(0.5, st.session_state["pdf_zoom"] - 0.2)

        zoom = st.session_state["pdf_zoom"]

        # 渲染 PDF
        st.pdf(pdf_path, height=850)

        # 用 CSS 动态缩放 PDF 的 iframe，并启用横向滚动容器
        # iframe 由 st.pdf 生成，通过 transform: scale() 缩放
        # 父容器设 overflow-x:auto，配合 Shift+滚轮横向滚动
        pdf_control_html = f"""
        <style>
            /* PDF iframe 父容器设为可横向滚动 */
            div[data-testid="column"]:nth-child(2) div[style*="overflow"] {{
                overflow-x: auto !important;
                overflow-y: auto !important;
            }}
            /* 缩放 st.pdf 生成的 iframe */
            div[data-testid="column"]:nth-child(2) iframe {{
                transform: scale({zoom}) !important;
                transform-origin: top left !important;
                width: {100 / zoom}% !important;
            }}
        </style>
        <script>
        (function() {{
            // Shift + 滚轮 → 横向滚动 PDF 容器
            const pdfCol = document.querySelector('div[data-testid="column"]:nth-child(2)');
            if (!pdfCol) return;
            if (pdfCol.dataset.shiftScrollBound === '1') return;
            pdfCol.dataset.shiftScrollBound = '1';

            pdfCol.addEventListener('wheel', function(e) {{
                if (e.shiftKey) {{
                    e.preventDefault();
                    // 找到可滚动的容器
                    let scrollable = pdfCol.querySelector('div[style*="overflow"]');
                    if (!scrollable) scrollable = pdfCol;
                    scrollable.scrollLeft += e.deltaY;
                }}
            }}, {{ passive: false }});
        }})();
        </script>
        """
        st.markdown(pdf_control_html, unsafe_allow_html=True)
