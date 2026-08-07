"""
LLM API 调用封装模块
封装对智谱 GLM-4.7-Flash 模型的 API 调用，使用 OpenAI 兼容接口。
扩展预留：后续可在此处添加重试机制、流式输出、多模型支持等功能。
"""

from typing import Optional

from openai import OpenAI

from core.config import settings


class LLMClient:
    """大模型 API 客户端，封装对智谱 GLM 系列的调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.GLM_API_KEY,
            base_url=settings.model_base_url,
        )
        self.model_name = settings.model_name

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        发送聊天请求并获取回复

        Args:
            messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            top_p: 核采样参数

        Returns:
            模型生成的回复文本
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or settings.model_temperature,
            max_tokens=max_tokens or settings.model_max_tokens,
            top_p=top_p or settings.model_top_p,
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        流式聊天接口，逐 chunk 产出内容

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: 核采样参数

        Yields:
            模型返回的文本片段
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=temperature or settings.model_temperature,
            max_tokens=max_tokens or settings.model_max_tokens,
            top_p=top_p or settings.model_top_p,
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