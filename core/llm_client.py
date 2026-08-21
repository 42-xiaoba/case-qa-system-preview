"""
LLM API 调用封装模块
封装对智谱 GLM-4.7-Flash 模型的 API 调用，使用 OpenAI 兼容接口。
内置：429 限流指数退避重试、混合推理模型 thinking 开关、流式输出。
"""

import time
from typing import Optional

from openai import OpenAI, RateLimitError

from core.config import settings

_MAX_RETRIES = 4          # 首次请求 + 3 次重试
_RETRY_BASE_DELAY = 3.0   # 首次退避秒数，之后指数递增（3s → 6s → 12s）


class LLMClient:
    """大模型 API 客户端，封装对智谱 GLM 系列的调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.GLM_API_KEY,
            base_url=settings.model_base_url,
        )
        self.model_name = settings.model_name

    @staticmethod
    def _thinking_extra_body() -> dict:
        """glm-4.7 系列为混合推理模型：思考 token 会挤占 max_tokens，
        导致非流式调用在思考阶段就被截断而返回空正文。默认关闭。"""
        return {
            "thinking": {
                "type": "enabled" if settings.model_thinking else "disabled"
            }
        }

    def _create_with_retry(self, max_attempts: int, **kwargs):
        """带指数退避的请求封装：免费模型高峰期频繁返回 429，
        OpenAI SDK 内置重试（2次、间隔极短）不足以穿透限流。
        429 仅在 create() 阶段抛出（流式调用此时还未产出任何 chunk），重试安全。"""
        for attempt in range(max_attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求并获取回复

        Args:
            messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            retries: 429 最大尝试次数，默认用模块常量（批量评测可传更大值）

        Returns:
            模型生成的回复文本
        """
        response = self._create_with_retry(
            retries if retries is not None else _MAX_RETRIES,
            model=self.model_name,
            messages=messages,
            temperature=temperature or settings.model_temperature,
            max_tokens=max_tokens or settings.model_max_tokens,
            top_p=top_p or settings.model_top_p,
            extra_body=self._thinking_extra_body(),
        )
        message = response.choices[0].message
        return message.content or ""

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        retries: Optional[int] = None,
    ):
        """
        流式聊天接口，逐 chunk 产出内容

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            retries: 429 限流重试次数（None 则用默认值）

        Yields:
            模型返回的文本片段
        """
        response = self._create_with_retry(
            retries if retries is not None else _MAX_RETRIES,
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=temperature or settings.model_temperature,
            max_tokens=max_tokens or settings.model_max_tokens,
            top_p=top_p or settings.model_top_p,
            extra_body=self._thinking_extra_body(),
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ---- 扩展预留：多轮对话历史管理 ----
    # def chat_with_history(self, messages, history=None, **kwargs):
    #     """带历史记录的对话（预留）"""
    #     full_messages = (history or []) + messages
    #     return self.chat(full_messages, **kwargs)


class VisionLLMClient:
    """视觉大模型 API 客户端，封装对智谱 GLM-4.6v-Flash 多模态模型的调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.GLM_V_API_KEY,
            base_url=settings.vision_model_base_url,
        )
        self.model_name = settings.vision_model_name

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """发送多模态聊天请求并获取回复"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or settings.vision_model_temperature,
            max_tokens=max_tokens or settings.vision_model_max_tokens,
            top_p=top_p or settings.vision_model_top_p,
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """流式多模态聊天接口，逐 chunk 产出内容"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=temperature or settings.vision_model_temperature,
            max_tokens=max_tokens or settings.vision_model_max_tokens,
            top_p=top_p or settings.vision_model_top_p,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# 全局单例
llm_client = LLMClient()

# 视觉客户端延迟创建（避免模块导入时密钥未就绪导致永久为 None）
_vision_llm_client = None


def get_vision_llm_client():
    """获取视觉客户端（延迟创建，每次调用都检查密钥是否可用）"""
    global _vision_llm_client
    if settings.vision_enabled:
        if _vision_llm_client is None:
            _vision_llm_client = VisionLLMClient()
        return _vision_llm_client
    return None