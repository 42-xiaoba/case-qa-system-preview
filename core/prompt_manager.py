"""
提示词管理模块
负责将 config.yaml 中的三层提示词与 case.txt 案例内容组装成完整的 System Prompt。
扩展预留：后续可在此处添加动态 prompt 模板、RAG 上下文注入等功能。
"""

from pathlib import Path

from core.config import settings

ROOT = Path(__file__).resolve().parent.parent


class PromptManager:
    """提示词管理器，负责组装和优化系统提示词"""

    def __init__(self):
        self.settings = settings

    # ---- 多视角回答 ----

    def build_perspective_instruction(self, perspective: str | None) -> str:
        """按视角 key 返回【回答视角设定】指令文本；无效或未传返回空串（不作注入）"""
        options = self.settings.perspectives_config.get("options", {}) or {}
        instruction = ((options.get(perspective) or {}).get("instruction") or "").strip()
        return instruction

    def perspective_options(self) -> list[tuple[str, str]]:
        """可选视角列表 [(key, 展示label)]，按 config.yaml 中的书写顺序"""
        options = self.settings.perspectives_config.get("options", {}) or {}
        return [(key, (opt or {}).get("label") or key) for key, opt in options.items()]

    def default_perspective(self) -> str:
        """默认视角 key：取 config 的 default，缺省时退回第一个选项"""
        cfg = self.settings.perspectives_config
        options = cfg.get("options", {}) or {}
        default = cfg.get("default")
        if default in options:
            return default
        return next(iter(options), "")

    def perspective_label(self, perspective: str | None) -> str:
        """视角 key → 展示 label（侧栏标题等 UI 处使用）"""
        options = self.settings.perspectives_config.get("options", {}) or {}
        return ((options.get(perspective) or {}).get("label")) or perspective or ""

    def build_system_prompt(self, perspective: str | None = None) -> str:
        """
        组装完整的系统提示词（System Prompt）
        将三层提示词 + 案例文本拼接为完整指令

        组装结构：
        [第一层：系统人设与角色]
        [第二层：核心业务规则]
        [回答视角设定（选择非默认视角时注入）]
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
        ]
        perspective_block = self.build_perspective_instruction(perspective)
        if perspective_block:
            sections += [
                "",
                "=" * 60,
                "【回答视角设定】",
                "=" * 60,
                "在完全遵守上述资料铁律与引用规范的前提下，按以下视角要求组织表达：",
                perspective_block,
            ]
        sections += [
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

    def build_vision_messages(
        self,
        user_query: str,
        image_data_url: str,
        history: list | None = None,
        perspective: str | None = None,
    ) -> list[dict]:
        """
        构建含图像的多模态消息列表，供视觉模型 API 调用

        Args:
            user_query: 用户当前输入的文本问题
            image_data_url: 图片的 base64 data URL（如 "data:image/png;base64,..."）
            history: 历史对话记录
            perspective: 回答视角 key（None = 默认视角）

        Returns:
            完整的多模态消息列表
        """
        system_prompt = self.build_system_prompt(perspective=perspective)
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话（如果有）
        if history:
            messages.extend(history)

        # 添加当前用户输入（多模态格式：文本 + 图片）
        user_content = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        messages.append({"role": "user", "content": user_content})
        return messages

    # ---- RAG 路径：预算制组装（Step 3）----
    # 注意：RAG 路径的 system prompt 不再注入 case.txt 全文，
    # 案例事实由检索块提供（避免 5 万字全文导致的 lost-in-the-middle 与高延迟）。
    # case.txt 全文注入仅保留给视觉路径 build_system_prompt()。

    def _tier0_card(self) -> str:
        """读取常备知识卡（Tier0），带字符预算截断"""
        cfg = self.settings.rag_config
        cap = int(cfg.get("budget", {}).get("tier0_chars", 1600))
        path = ROOT / cfg.get("tier0_card_path", "kb/tier0_card.md")
        if not path.exists():
            return ""
        card = path.read_text(encoding="utf-8").strip()
        return card[:cap]

    def build_rag_system_prompt(
        self,
        retrieved_docs: list[dict],
        history_summary: str | None = None,
        perspective: str | None = None,
    ) -> str:
        """
        组装 RAG 路径的系统提示词

        结构：人设 → 核心规则 → 视角设定（选择非默认视角时注入）→ Tier0 常备知识卡
        → 检索块（编号带来源）→ 历史摘要 → 输出格式
        """
        parts = [
            "=" * 60,
            "【系统人设与角色】",
            "=" * 60,
            self.settings.prompt_system_role,
            "",
            "=" * 60,
            "【核心业务规则】",
            "=" * 60,
            self.settings.prompt_core_rules,
        ]
        perspective_block = self.build_perspective_instruction(perspective)
        if perspective_block:
            parts += [
                "",
                "=" * 60,
                "【回答视角设定】",
                "=" * 60,
                "在完全遵守上述资料铁律与引用规范的前提下，按以下视角要求组织表达：",
                perspective_block,
            ]
        tier0 = self._tier0_card()
        if tier0:
            parts += [
                "",
                "=" * 60,
                "【常备知识卡：案例核心事实（最高可信度，可直接引用）】",
                "=" * 60,
                tier0,
            ]
        parts += [
            "",
            "=" * 60,
            "【检索到的参考资料（与当前问题最相关的片段，按相关性排序）】",
            "=" * 60,
            retrieved_docs if isinstance(retrieved_docs, str) else self._format_docs(retrieved_docs),
            "",
            "=" * 60,
            "【输出格式要求】",
            "=" * 60,
            self.settings.prompt_output_format,
        ]
        # 历史摘要插在输出格式之前，靠近对话历史位置
        if history_summary:
            parts.insert(-2, "\n" + "=" * 60 + "\n【历史对话摘要（较早对话的要点，供理解上下文）】\n" + "=" * 60 + "\n" + history_summary)
        return "\n".join(parts)

    @staticmethod
    def _format_docs(docs: list[dict]) -> str:
        """检索块编号格式化，带来源标注便于模型引用"""
        if not docs:
            return "（未检索到相关资料，请依据常备知识卡作答，或说明资料未涉及）"
        lines = []
        for i, doc in enumerate(docs):
            lines.append(f"[{i + 1}]【{doc['source']}·{doc['section']}】{doc['content']}")
        return "\n\n".join(lines)

    def build_messages_with_rag(
        self,
        user_query: str,
        retrieved_docs: list[dict],
        history: list | None = None,
        history_summary: str | None = None,
        perspective: str | None = None,
    ) -> list[dict]:
        """
        带检索结果的完整消息构建（预算制组装）

        预算裁剪顺序：
        1. 检索块超 context_chars → 从相关性最低的尾部丢弃
        2. 历史对话超 history_chars → 丢弃最早的对话
        3. 总量超 total_chars → 继续裁剪历史（规则与知识卡永不裁剪）

        Args:
            user_query: 用户当前输入
            retrieved_docs: 检索到的知识块（按相关性降序）
            history: 历史对话 [{"role","content"},...]
            history_summary: 较早对话的滚动摘要（P1 记忆压缩产出）
            perspective: 回答视角 key（None = 默认视角）

        Returns:
            完整消息列表
        """
        budget = self.settings.rag_config.get("budget", {})
        context_cap = int(budget.get("context_chars", 4200))
        history_cap = int(budget.get("history_chars", 2200))
        summary_cap = int(budget.get("summary_chars", 600))
        total_cap = int(budget.get("total_chars", 12000))

        # 1. 检索块预算：docs 已按相关性降序，超限从尾部（最不重要）丢弃
        kept_docs = []
        used = 0
        for doc in retrieved_docs:
            block_len = len(doc["content"]) + len(doc["source"]) + len(doc["section"]) + 8
            if used + block_len > context_cap and kept_docs:
                break
            kept_docs.append(doc)
            used += block_len

        if history_summary and len(history_summary) > summary_cap:
            history_summary = history_summary[:summary_cap]

        # 2. 历史预算：从最新往旧保留
        kept_history: list[dict] = []
        h_used = 0
        if history:
            for msg in reversed(history):
                n = len(msg.get("content", "") or "")
                if h_used + n > history_cap and kept_history:
                    break
                kept_history.insert(0, msg)
                h_used += n

        system_prompt = self.build_rag_system_prompt(kept_docs, history_summary, perspective)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(kept_history)
        messages.append({"role": "user", "content": user_query})

        # 3. 总量兜底：从最旧的历史消息开始丢弃
        while len(system_prompt) + h_used + len(user_query) > total_cap and kept_history:
            removed = kept_history.pop(0)
            h_used -= len(removed.get("content", "") or "")
        if kept_history:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(kept_history)
            messages.append({"role": "user", "content": user_query})
        return messages


# 全局单例
prompt_manager = PromptManager()