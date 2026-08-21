"""
Streamlit 前端界面
提供智能案例问答系统的交互界面。
支持两种运行模式：
  - API 模式：依赖 FastAPI 后端（本地开发）
  - 直连模式：直接调用 LLM API（Streamlit Cloud 部署）
"""

import base64
from pathlib import Path

import httpx
import streamlit as st

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
for _key in ("GLM_API_KEY", "GLM_V_API_KEY"):
    try:
        _val = st.secrets[_key]
        if _val:
            _os.environ[_key] = str(_val).strip()
    except Exception as e:
        _secrets_errors.append(f"{_key}: {e}")

from core.llm_client import llm_client, get_vision_llm_client
from core.memory import prepare as memory_prepare
from core.pipeline import build_answer_messages_routed
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


_WAITING_MSG = "⏳ 正在思考你的问题，可能会有些慢，请不要着急...\n\n"

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


def send_chat_request_stream_direct(query: str, history: list | None = None):
    """直连模式流式生成器（记忆压缩 → 路由 → 检索 → 预算制组装）"""
    yield _WAITING_MSG
    windowed, new_summary = memory_prepare(
        history or [], st.session_state.get("history_summary")
    )
    st.session_state["history_summary"] = new_summary
    messages, _docs, _route = build_answer_messages_routed(
        query,
        history=windowed,
        history_summary=new_summary or None,
    )
    try:
        for chunk in llm_client.chat_stream(messages):
            yield chunk
    except Exception as e:
        yield f"\n\n[错误] 模型调用失败: {e}"


def send_chat_request_stream_api(query: str, history: list | None = None):
    """API 模式流式生成器"""
    yield _WAITING_MSG
    payload = {"query": query, "history": history or []}
    try:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/api/chat/stream",
                json=payload,
                timeout=60,
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


def send_vision_chat_stream_direct(query: str, image_data_url: str, history: list | None = None):
    """直连模式视觉流式生成器"""
    yield _WAITING_MSG
    messages = prompt_manager.build_vision_messages(
        user_query=query, image_data_url=image_data_url, history=history or []
    )
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
        "智能案例问答系统 v0.3.0<br>团队成员：<br>卜天伊 冯思杰 等</div>",
        unsafe_allow_html=True,
    )


# ==================== 主界面 ====================

load_custom_css()

left_col, right_col = st.columns([0.6, 0.4], gap="medium", vertical_alignment="top")


# ==================== 左栏：AI 问答 ====================

with left_col:
    st.caption("💬 案例问答 · GLM-4.7-Flash")

    # 可滚动聊天容器，固定像素高度
    chat_container = st.container(height=750)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                render_message_content(msg["content"])

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

            with st.chat_message("assistant"):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["content"] != GREETING_MESSAGE  # 问候语不作为上下文发送
                ]

                placeholder = st.empty()
                base_wait_msg = "⏳ 正在思考你的问题，可能会有些慢，请不要着急"

                if pending_image and get_vision_llm_client() is not None:
                    # 有图片：走视觉模型
                    if st.session_state.api_healthy:
                        stream = send_vision_chat_stream_api(prompt, pending_image, history)
                    else:
                        stream = send_vision_chat_stream_direct(prompt, pending_image, history)
                else:
                    # 无图片：走文本模型（原有逻辑）
                    if st.session_state.api_healthy:
                        stream = send_chat_request_stream_api(prompt, history)
                    else:
                        stream = send_chat_request_stream_direct(prompt, history)

                # 后台线程：跳过 _WAITING_MSG 占位，获取第一个真实 chunk（这里会阻塞等待 API 响应）
                import threading
                import queue
                import time

                first_real_chunk_q = queue.Queue()

                def fetch_first_real_chunk():
                    try:
                        next(stream)  # 消费掉 _WAITING_MSG 占位（立即返回）
                        first = next(stream)  # 等待 API 第一个真实 token（阻塞）
                        first_real_chunk_q.put(("chunk", first))
                    except StopIteration:
                        first_real_chunk_q.put(("empty", None))
                    except Exception as e:
                        first_real_chunk_q.put(("error", str(e)))

                fetch_thread = threading.Thread(target=fetch_first_real_chunk, daemon=True)
                fetch_thread.start()

                # 主线程：播放省略号动画，直到第一个真实 chunk 到达
                dots = 0
                status = None
                first_chunk = None
                while True:
                    dots = (dots % 3) + 1
                    placeholder.write(base_wait_msg + "." * dots)
                    try:
                        status, first_chunk = first_real_chunk_q.get(timeout=1.0)
                        break
                    except queue.Empty:
                        continue

                fetch_thread.join(timeout=0.5)

                if status == "error":
                    response_text = f"[错误] {first_chunk}"
                    placeholder.write("✅ 已完成思考\n\n" + response_text)
                elif status == "empty":
                    response_text = "（未获取到回复）"
                    placeholder.write("✅ 已完成思考\n\n" + response_text)
                else:
                    # 第一个真实 chunk 到达，切换为流式回复显示
                    response_text = first_chunk
                    placeholder.write("✅ 已完成思考\n\n" + response_text)
                    for chunk in stream:
                        response_text += chunk
                        placeholder.write("✅ 已完成思考\n\n" + response_text)

                    if not response_text:
                        response_text = "（未获取到回复）"
                        placeholder.write("✅ 已完成思考\n\n" + response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


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
