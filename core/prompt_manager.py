"""
提示词管理模块
负责将 config.yaml 中的三层提示词与 case.txt 案例内容组装成完整的 System Prompt。
扩展预留：后续可在此处添加动态 prompt 模板、RAG 上下文注入等功能。
"""

from core.config import settings


class PromptManager:
    """提示词管理器，负责组装和优化系统提示词"""

    def __init__(self):
        self.settings = settings

    def build_system_prompt(self) -> str:
        """
        组装完整的系统提示词（System Prompt）
        将三层提示词 + 案例文本拼接为完整指令

        组装结构：
        [第一层：系统人设与角色]
        [第二层：核心业务规则]
        [案例文本全文]
        [第三层：输出格式要求]
        """
        sections = [
            "=" * 60,
            "【系统人设与角色】",
            "=" * 60,
            self.settings.prompt_system_role,
            "",
            "=" * 60,
            "【核心业务规则与案例知识】",
            "=" * 60,
            self.settings.prompt_core_rules,
            "",
            "=" * 60,
            "【以下为本次分析的完整案例文本，请仔细阅读】",
            "=" * 60,
            self.settings.case_text,
            "",
            "=" * 60,
            "【输出格式要求】",
            "=" * 60,
            self.settings.prompt_output_format,
        ]
        return "\n".join(sections)

    def build_messages(self, user_query: str, history: list | None = None) -> list[dict]:
        """
        构建完整的消息列表，供 API 调用

        Args:
            user_query: 用户当前输入
            history: 历史对话记录，格式为 [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            完整的消息列表
        """
        system_prompt = self.build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话（如果有）
        if history:
            messages.extend(history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_query})
        return messages

    # ---- 扩展预留：RAG 上下文注入 ----
    # def build_messages_with_rag(self, user_query, retrieved_docs, history=None):
    #     """带 RAG 检索结果的 prompt 构建（预留）"""
    #     context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    #     system_prompt = self.build_system_prompt()
    #     augmented_prompt = (
    #         f"{system_prompt}\n\n"
    #         f"【检索到的相关参考资料】\n{context}\n\n"
    #         f"【用户问题】\n{user_query}"
    #     )
    #     messages = [{"role": "system", "content": augmented_prompt}]
    #     if history:
    #         messages.extend(history)
    #     messages.append({"role": "user", "content": user_query})
    #     return messages


# 全局单例
prompt_manager = PromptManager()