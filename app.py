"""
FastAPI 后端服务
提供聊天问答 API 接口，供 Streamlit 前端调用。
扩展预留：已预留文件上传、RAG 检索等接口位置。
"""

import json
import logging

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.llm_client import llm_client, get_vision_llm_client
from core.memory import prepare as memory_prepare
from core.pipeline import build_answer_messages_routed
from core.prompt_manager import prompt_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="智能案例问答系统 API",
    description="基于 GLM-4.7-Flash 的案例问答后端服务",
    version="1.0.0",
)

# CORS 配置，允许 Streamlit 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., description="用户输入的问题")
    history: list[dict] | None = Field(default=None, description="历史对话记录")


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str = Field(..., description="模型回复")
    success: bool = Field(default=True, description="请求是否成功")
    sources: list[str] | None = Field(default=None, description="本次注入的检索块章节来源")
    route: dict | None = Field(default=None, description="路由分类信息（类型+改写后查询）")


class VisionChatRequest(BaseModel):
    """视觉聊天请求（含图片）"""
    query: str = Field(..., description="用户输入的问题")
    image: str = Field(..., description="图片的 base64 data URL，如 data:image/png;base64,...")
    history: list[dict] | None = Field(default=None, description="历史对话记录")


# ==================== API 接口 ====================


@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "智能案例问答系统运行正常"}


@app.get("/api/pdf", tags=["系统"])
async def get_pdf():
    """
    提供 case_report.pdf 文件下载/预览
    前端通过 iframe/embed 引用此 URL 即可显示 PDF
    """
    pdf_path = Path(__file__).resolve().parent / "case_report.pdf"
    if not pdf_path.exists():
        return {"error": "PDF 文件不存在", "path": str(pdf_path)}
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="case_report.pdf",
        headers={"Content-Disposition": "inline; filename=case_report.pdf"},
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["问答"])
async def chat(request: ChatRequest):
    """
    聊天问答接口（非流式）
    完整管线：记忆压缩 → 路由分类/追问改写 → 检索 → 预算制组装 → 生成
    """
    try:
        windowed, summary = memory_prepare(request.history or [], None)
        messages, docs, route = build_answer_messages_routed(
            request.query,
            history=windowed,
            history_summary=summary or None,
        )
        reply = llm_client.chat(messages)
        return ChatResponse(
            reply=reply,
            sources=[f"tier{d.get('tier')}·{d.get('section', '')}" for d in docs],
            route={"type": route.get("type"), "rewritten_query": route.get("rewritten_query")},
        )
    except Exception as e:
        logger.error(f"聊天请求失败: {e}")
        return ChatResponse(
            reply=f"抱歉，回答生成时出现错误：{str(e)}。请稍后重试。",
            success=False,
        )


@app.post("/api/chat/stream", tags=["问答"])
async def chat_stream(request: ChatRequest):
    """
    聊天问答接口（流式）
    接收用户问题，以 SSE 格式流式返回模型回复
    """
    try:
        windowed, summary = memory_prepare(request.history or [], None)
        messages, _docs, _route = build_answer_messages_routed(
            request.query,
            history=windowed,
            history_summary=summary or None,
        )
    except Exception as e:
        logger.error(f"构建消息失败: {e}")

        async def error_gen():
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(error_gen(), media_type="text/plain")

    def generate():
        """同步生成器，逐 chunk 产出文本"""
        try:
            for chunk in llm_client.chat_stream(messages):
                yield chunk
        except Exception as e:
            logger.error(f"流式请求失败: {e}")
            yield f"\n\n[错误] {e}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/chat/vision/stream", tags=["问答"])
async def chat_vision_stream(request: VisionChatRequest):
    """
    视觉聊天问答接口（流式）
    接收用户问题 + base64 图片，以流式格式返回视觉模型回复
    """
    vision_llm_client = get_vision_llm_client()
    if vision_llm_client is None:
        async def disabled_gen():
            yield json.dumps(
                {"error": "视觉功能未启用，请在 .env 中配置 GLM_V_API_KEY"},
                ensure_ascii=False,
            ) + "\n"
        return StreamingResponse(disabled_gen(), media_type="text/plain")

    try:
        messages = prompt_manager.build_vision_messages(
            user_query=request.query,
            image_data_url=request.image,
            history=request.history,
        )
    except Exception as e:
        logger.error(f"构建视觉消息失败: {e}")

        async def error_gen():
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(error_gen(), media_type="text/plain")

    def generate():
        """同步生成器，逐 chunk 产出文本"""
        try:
            for chunk in vision_llm_client.chat_stream(messages):
                yield chunk
        except Exception as e:
            logger.error(f"视觉流式请求失败: {e}")
            yield f"\n\n[错误] {e}"

    return StreamingResponse(generate(), media_type="text/plain")


# ==================== 扩展预留接口 ====================

# ---- 文件上传（预留） ----
# @app.post("/api/upload", tags=["扩展"])
# async def upload_file(file: UploadFile = File(...)):
#     """上传文件（预留：用于后续 RAG 知识库扩展）"""
#     pass

# ---- 清空对话（预留） ----
# @app.post("/api/clear", tags=["扩展"])
# async def clear_session(session_id: str = Body(...)):
#     """清空对话历史（预留）"""
#     pass


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    from core.config import settings

    uvicorn.run(
        "app:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True,
    )