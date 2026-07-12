"""
Streamlit 前端界面
提供智能案例问答系统的交互界面。
支持两种运行模式：
  - API 模式：依赖 FastAPI 后端（本地开发）
  - 直连模式：直接调用 LLM API（Streamlit Cloud 部署）
扩展预留：已预留 CSS 主题美化、侧边栏功能扩展接口。
"""

import base64
from pathlib import Path

import httpx
import streamlit as st

# 核心模块（直连模式使用）
from core.llm_client import llm_client
from core.prompt_manager import prompt_manager

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="智能案例问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 自定义 CSS ====================
# 侧边栏悬停弹出 + 页面布局样式

CUSTOM_CSS = """
<style>
    /* ========== 侧边栏悬停弹出 ========== */
    /* 默认收起状态：60px 宽度，隐藏溢出内容 */
    section[data-testid="stSidebar"] {
        width: 60px !important;
        min-width: 60px !important;
        max-width: 60px !important;
        transition: width 0.3s ease, min-width 0.3s ease, max-width 0.3s ease !important;
        overflow: hidden !important;
    }
    /* 悬停展开状态：250px 宽度，显示滚动条 */
    section[data-testid="stSidebar"]:hover {
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        overflow: auto !important;
    }
    /* 侧边栏内部容器边距归零，节省空间 */
    section[data-testid="stSidebar"] > div:first-child {
        padding: 0.5rem !important;
    }
    /* 侧边栏中所有元素默认半透明，悬停时恢复 */
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
    /* 汉堡图标始终可见 */
    .sidebar-hamburger {
        font-size: 24px;
        text-align: center;
        padding: 8px 0;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    /* ========== 页面头部 ========== */
    .main-header {
        text-align: center;
        padding: 0.5rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #1f77b4;
        font-size: 1.6rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #666;
        font-size: 0.85rem;
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


# 等待提示消息（流式输出时先展示，不会被保存到历史记录）
_WAITING_MSG = "⏳ 正在思考你的问题，可能会有些慢，请不要着急...\n\n"


def send_chat_request_stream_direct(query: str, history: list | None = None):
    """
    直连模式：直接调用 LLM 的流式生成器
    不依赖 FastAPI 后端，适用于 Streamlit Cloud 部署

    Args:
        query: 用户输入
        history: 历史对话记录

    Yields:
        文本片段（等待提示 → 实际回复）
    """
    yield _WAITING_MSG
    messages = prompt_manager.build_messages(user_query=query, history=history or [])
    try:
        for chunk in llm_client.chat_stream(messages):
            yield chunk
    except Exception as e:
        yield f"\n\n[错误] 模型调用失败: {e}"


def send_chat_request_stream_api(query: str, history: list | None = None):
    """
    API 模式：通过 FastAPI 后端获取流式回复

    Args:
        query: 用户输入
        history: 历史对话记录

    Yields:
        文本片段（等待提示 → 实际回复）
    """
    yield _WAITING_MSG
    payload = {
        "query": query,
        "history": history or [],
    }
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
    st.session_state.messages = []

if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False


# ==================== 侧边栏 ====================

with st.sidebar:
    # 汉堡图标（始终可见，标记侧边栏收起位置）
    st.markdown(
        '<div class="sidebar-hamburger">☰</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 💬 智能案例问答系统")
    st.markdown("---")

    # 系统状态
    st.markdown("### 系统状态")
    api_ok = check_api_health()
    st.session_state.api_healthy = api_ok
    if api_ok:
        st.success("✅ API 模式（FastAPI 后端运行中）")
    else:
        st.warning("🔄 直连模式（直接调用 LLM API）")
        st.caption("本地开发请同时运行 `python app.py`")

    st.markdown("---")

    # ---- 扩展预留：文件上传功能 ----
    st.markdown("### 📁 文件上传（预留）")
    st.markdown("*此功能即将上线，敬请期待*")
    # uploaded_file = st.file_uploader("上传案例文件", type=["txt", "pdf", "docx"])
    # if uploaded_file:
    #     st.success(f"已上传：{uploaded_file.name}")

    st.markdown("---")

    # ---- 扩展预留：功能按钮 ----
    st.markdown("### ⚙️ 功能")

    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ---- 扩展预留：更多功能入口 ----
    # st.button("📊 查看分析报告", use_container_width=True)
    # st.button("🔍 高级检索", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
    <div style="font-size: 0.8rem; color: #999; text-align: center;">
        智能案例问答系统 v1.0<br>
        技术支持：GLM-4.7-Flash
    </div>
    """,
        unsafe_allow_html=True,
    )


# ==================== 主界面（左右两栏） ====================

load_custom_css()

# 页面标题
st.markdown(
    '<div class="main-header"><h1>📚 智能案例问答系统</h1>'
    "<p>基于武侯区报表通与社管通案例的智能问答助手</p></div>",
    unsafe_allow_html=True,
)

# 左右两栏布局
left_col, right_col = st.columns([0.6, 0.4], gap="large")


# ==================== 左栏：AI 问答界面 ====================

with left_col:
    # 显示聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("请输入你的问题，例如：这个案例的研究意义是什么？"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 根据模式选择流式生成器
        with st.chat_message("assistant"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            if st.session_state.api_healthy:
                # API 模式：通过 FastAPI 后端
                full_text = st.write_stream(
                    send_chat_request_stream_api(prompt, history)
                )
            else:
                # 直连模式：直接调用 LLM
                full_text = st.write_stream(
                    send_chat_request_stream_direct(prompt, history)
                )

        # 去除等待提示前缀，只保存实际回复到历史记录
        if full_text.startswith(_WAITING_MSG):
            reply = full_text[len(_WAITING_MSG):]
        else:
            reply = full_text
        st.session_state.messages.append({"role": "assistant", "content": reply})


# ==================== 右栏：案例报告 PDF 预览 ====================

with right_col:
    st.markdown("### 📄 案例报告")

    pdf_path = Path(__file__).resolve().parent / "case_report.pdf"

    if not pdf_path.exists():
        st.error("❌ 未找到 `case_report.pdf`，请放入项目根目录")
    else:
        if st.session_state.api_healthy:
            # API 模式：通过后端 iframe 预览
            try:
                with httpx.Client() as client:
                    resp = client.get(f"{API_BASE_URL}/api/pdf", timeout=5)
                    if resp.status_code == 200 and resp.headers.get("content-type") == "application/pdf":
                        pdf_iframe = f"""
                        <iframe
                            src="{API_BASE_URL}/api/pdf"
                            width="100%"
                            height="700px"
                            style="border:1px solid #e0e0e0; border-radius:8px;"
                        >
                            您的浏览器不支持 iframe 预览，请
                            <a href="{API_BASE_URL}/api/pdf" target="_blank">直接打开 PDF</a>
                        </iframe>
                        """
                        st.markdown(pdf_iframe, unsafe_allow_html=True)
                        st.caption(f"💡 如果预览空白，请 [直接打开 PDF]({API_BASE_URL}/api/pdf)")
                    else:
                        st.error(f"❌ PDF 服务异常: HTTP {resp.status_code}")
            except httpx.ConnectError:
                st.error("❌ 后端服务未连接，PDF 无法加载")
        else:
            # 直连模式：用 PDF 下载按钮替代
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 下载 PDF 查看",
                    data=pdf_bytes,
                    file_name="case_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.info("直连模式下 PDF 预览不可用，请点击上方按钮下载查看。")
            except Exception as e:
                st.error(f"❌ PDF 读取失败: {e}")


# ==================== 扩展预留：底部信息栏 ====================
# st.markdown("---")
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown("**当前案例**：武侯区报表通与社管通")
# with col2:
#     st.markdown("**模型**：GLM-4.7-Flash")
# with col3:
#     st.markdown("**状态**：运行中")