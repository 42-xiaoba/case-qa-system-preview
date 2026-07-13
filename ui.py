"""
Streamlit 前端界面
提供智能案例问答系统的交互界面。
支持两种运行模式：
  - API 模式：依赖 FastAPI 后端（本地开发）
  - 直连模式：直接调用 LLM API（Streamlit Cloud 部署）
"""

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
try:
    if "GLM_API_KEY" in st.secrets:
        import os
        os.environ["GLM_API_KEY"] = st.secrets["GLM_API_KEY"]
except Exception:
    pass

from core.llm_client import llm_client
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
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] .stAlert {
        opacity: 0;
        transition: opacity 0.25s ease 0.05s;
        pointer-events: none;
    }
    section[data-testid="stSidebar"]:hover .stMarkdown,
    section[data-testid="stSidebar"]:hover .stButton,
    section[data-testid="stSidebar"]:hover .stAlert {
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
</style>
"""


def load_custom_css():
    """加载自定义 CSS"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== 后端 API 配置 ====================

API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """检查后端服务是否正常运行"""
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE_URL}/api/health", timeout=5)
            return resp.json().get("status") == "ok"
    except Exception:
        return False


_WAITING_MSG = "⏳ 正在思考你的问题，可能会有些慢，请不要着急...\n\n"


def send_chat_request_stream_direct(query: str, history: list | None = None):
    """直连模式流式生成器"""
    yield _WAITING_MSG
    messages = prompt_manager.build_messages(user_query=query, history=history or [])
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


# ==================== 会话状态初始化 ====================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是智能案例问答助手，请随时向我提问关于案例的问题。"}
    ]

if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False


# ==================== 侧边栏 ====================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-hamburger">☰</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## 💬 智能案例问答系统")
    st.markdown("---")
    st.markdown("### 系统状态")
    api_ok = check_api_health()
    st.session_state.api_healthy = api_ok
    if api_ok:
        st.success("✅ API 模式（FastAPI 后端运行中）")
    else:
        st.warning("🔄 直连模式（直接调用 LLM API）")
        st.caption("本地开发请同时运行 `python app.py`")
    st.markdown("---")
    st.markdown("### ⚙️ 功能")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是智能案例问答助手，请随时向我提问关于案例的问题。"}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown(
        '<div style="font-size: 0.8rem; color: #999; text-align: center;">'
        "智能案例问答系统 v0.1.0<br>技术支持：GLM-4.7-Flash</div>",
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
                st.write(msg["content"])

    if prompt := st.chat_input("请输入你的问题，例如：这个案例的研究意义是什么？"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                placeholder = st.empty()
                placeholder.write("⏳ 正在思考你的问题，可能会有些慢，请不要着急...")

                if st.session_state.api_healthy:
                    stream = send_chat_request_stream_api(prompt, history)
                else:
                    stream = send_chat_request_stream_direct(prompt, history)

                try:
                    next(stream)
                except StopIteration:
                    pass

                response_text = ""
                for chunk in stream:
                    response_text += chunk
                    placeholder.write("✅ 已完成思考\n\n" + response_text)

                if not response_text:
                    response_text = "（未获取到回复）"
                    placeholder.write("✅ 已完成思考\n\n" + response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


# ==================== 右栏：PDF 预览 ====================

with right_col:
    pdf_path = Path(__file__).resolve().parent / "case_report.pdf"

    if not pdf_path.exists():
        st.error("❌ 未找到 `case_report.pdf`，请放入项目根目录")
    else:
        # st.pdf 原生渲染，固定 800px 高度
        st.pdf(pdf_path, height=850)
