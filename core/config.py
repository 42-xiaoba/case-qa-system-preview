"""
配置加载模块
负责读取 config.yaml 和密钥信息，提供全局配置访问接口。
密钥来源：.env 文件（本地开发）；ui.py 启动时会将 st.secrets 注入环境变量（Streamlit Cloud）。
扩展预留：后续可在此处添加 RAG、主题等配置的加载逻辑。
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """应用配置类，单例模式加载所有配置"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """加载配置"""
        # 加载 .env 文件（本地开发）
        env_path = ROOT_DIR / ".env"
        load_dotenv(env_path)

        # 加载 config.yaml
        config_path = ROOT_DIR / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 读取 API Key
        # 优先从环境变量读取（ui.py 启动时已将 st.secrets 注入环境变量），
        # 环境变量为空时兜底从 st.secrets 直接读取（防止注入失败）
        self.GLM_API_KEY = self._read_secret("GLM_API_KEY")
        if not self.GLM_API_KEY:
            raise ValueError(
                "GLM_API_KEY 未配置。\n"
                "  - 本地开发：在 .env 文件中设置 GLM_API_KEY\n"
                "  - Streamlit Cloud：在 Secrets 管理页面设置 GLM_API_KEY"
            )

        # 视觉模型 API Key 改为动态读取（不固化在单例中）
        # 原因：Streamlit Cloud 首次启动时 st.secrets 可能未就绪，
        # 若在 _load() 时固化，后续 rerun 即使读到密钥也无法恢复
        self._glm_v_api_key_cache = None

        # 读取案例文本
        self._case_text = self._load_case_text()

    def _load_case_text(self) -> str:
        """加载案例文本"""
        case_path = ROOT_DIR / "case.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"案例文件不存在: {case_path}")
        with open(case_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _read_secret(key: str) -> str:
        """
        读取密钥：优先从环境变量，环境变量为空时兜底从 st.secrets 读取。
        这样即使 ui.py 的环境变量注入失败，也能从 st.secrets 直接获取。
        """
        value = os.getenv(key, "")
        if value:
            return value
        # 兜底：尝试从 Streamlit secrets 直接读取
        try:
            import streamlit as st
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        return ""

    # ---- 以下为便捷属性 ----

    @property
    def case_text(self) -> str:
        """获取完整案例文本"""
        return self._case_text

    @property
    def GLM_V_API_KEY(self) -> str:
        """视觉模型 API Key（动态读取，不固化在单例中）"""
        # 每次访问都重新读取，确保 Streamlit Cloud rerun 后能拿到最新值
        return self._read_secret("GLM_V_API_KEY")

    @property
    def config(self) -> dict:
        """获取原始配置字典"""
        return self._config

    # ---- 提示词相关 ----

    @property
    def prompt_system_role(self) -> str:
        """获取系统人设提示词"""
        return self._config["prompt"]["system_role"]

    @property
    def prompt_core_rules(self) -> str:
        """获取核心业务规则提示词"""
        return self._config["prompt"]["core_rules"]

    @property
    def prompt_output_format(self) -> str:
        """获取输出格式要求提示词"""
        return self._config["prompt"]["output_format"]

    # ---- 多视角回答 ----

    @property
    def perspectives_config(self) -> dict:
        """获取多视角回答配置（default 默认视角 + options 各视角的 label/instruction）"""
        return self._config.get("perspectives", {}) or {}

    # ---- 模型配置 ----

    @property
    def model_name(self) -> str:
        return self._config["model"]["name"]

    @property
    def model_base_url(self) -> str:
        return self._config["model"]["base_url"]

    @property
    def model_temperature(self) -> float:
        return self._config["model"]["temperature"]

    @property
    def model_max_tokens(self) -> int:
        return self._config["model"]["max_tokens"]

    @property
    def model_top_p(self) -> float:
        return self._config["model"]["top_p"]

    @property
    def model_thinking(self) -> bool:
        """是否开启深度思考（glm-4.7 系列为混合推理模型，默认关闭以保证非流式调用有正文输出）"""
        return bool(self._config.get("model", {}).get("thinking", False))

    @property
    def model_concurrency(self) -> int:
        """全局并发闸门上限：智谱免费档按并发数限流，超过的请求排队而非报错"""
        try:
            return max(1, int(self._config.get("model", {}).get("concurrency", 1)))
        except Exception:
            return 1

    @property
    def glm_api_keys(self) -> list[str]:
        """主模型 API Key 列表（多 Key 轮换）：除 GLM_API_KEY 外还识别
        GLM_API_KEY_2 ~ GLM_API_KEY_5，每把 Key 拥有独立的限流配额"""
        keys: list[str] = []
        names = ["GLM_API_KEY"] + [f"GLM_API_KEY_{n}" for n in range(2, 6)]
        for name in names:
            value = self._read_secret(name)
            if value and value not in keys:
                keys.append(value)
        return keys or [self.GLM_API_KEY]

    # ---- 视觉模型配置 ----

    @property
    def vision_enabled(self) -> bool:
        """视觉功能是否可用（密钥已配置即启用）"""
        return bool(self.GLM_V_API_KEY)

    @property
    def vision_model_name(self) -> str:
        return self._config["vision_model"]["name"]

    @property
    def vision_model_base_url(self) -> str:
        return self._config["vision_model"]["base_url"]

    @property
    def vision_model_temperature(self) -> float:
        return self._config["vision_model"]["temperature"]

    @property
    def vision_model_max_tokens(self) -> int:
        return self._config["vision_model"]["max_tokens"]

    @property
    def vision_model_top_p(self) -> float:
        return self._config["vision_model"]["top_p"]

    # ---- 模型降级链配置 ----

    @property
    def fallback_enabled(self) -> bool:
        """模型降级链是否启用（主模型受限/超时时自动切换备用模型）"""
        return bool(self._config.get("fallback", {}).get("enabled", False))

    @property
    def fallback_openrouter_model(self) -> str:
        """OpenRouter 备用模型名称"""
        return self._config.get("fallback", {}).get("openrouter", {}).get("name", "")

    @property
    def fallback_openrouter_base_url(self) -> str:
        """OpenRouter 接口地址"""
        return self._config.get("fallback", {}).get("openrouter", {}).get(
            "base_url", "https://openrouter.ai/api/v1"
        )

    @property
    def fallback_openrouter_api_key(self) -> str:
        """OpenRouter API Key（动态读取；兼容 .env 中的历史拼写 OEPNROUTER_API_KEY
        与标准拼写 OPENROUTER_API_KEY）"""
        cfg = self._config.get("fallback", {}).get("openrouter", {})
        env_names = [cfg.get("api_key_env"), "OEPNROUTER_API_KEY", "OPENROUTER_API_KEY"]
        for name in env_names:
            if not name:
                continue
            value = self._read_secret(name)
            if value:
                return value
        return ""

    @property
    def vision_api_keys(self) -> list[str]:
        """视觉模型 API Key 列表（多 Key 轮换）：识别 GLM_V_API_KEY 与 GLM_V_API_KEY_2，
        每把 Key 拥有独立的限流配额"""
        keys: list[str] = []
        for name in ["GLM_V_API_KEY", "GLM_V_API_KEY_2"]:
            value = self._read_secret(name)
            if value and value not in keys:
                keys.append(value)
        return keys or [self.GLM_V_API_KEY]

    @property
    def ultimate_model(self) -> str:
        """末位兜底模型名称（glm-4.6v，前序模型全部失效才调用）"""
        return self._config.get("fallback", {}).get("ultimate", {}).get("name", "glm-4.6v")

    @property
    def ultimate_base_url(self) -> str:
        """末位兜底模型接口地址（默认与主模型同为智谱接口）"""
        cfg = self._config.get("fallback", {}).get("ultimate", {})
        return cfg.get("base_url") or self.model_base_url

    @property
    def ultimate_api_key(self) -> str:
        """末位兜底模型 API Key（动态读取；优先 config 指定变量名，
        兼容标准拼写 ULTIMATE_API_KEY 与 .env 现用的 GLM_V_API_KEY_3）"""
        cfg = self._config.get("fallback", {}).get("ultimate", {})
        env_names = [cfg.get("api_key_env"), "ULTIMATE_API_KEY", "GLM_V_API_KEY_3"]
        for name in env_names:
            if not name:
                continue
            value = self._read_secret(name)
            if value:
                return value
        return ""

    @property
    def fallback_sensenova_model(self) -> str:
        """SenseNova 备用模型名称"""
        return self._config.get("fallback", {}).get("sensenova", {}).get("name", "")

    @property
    def fallback_sensenova_base_url(self) -> str:
        """SenseNova OpenAI 兼容接口地址"""
        cfg = self._config.get("fallback", {}).get("sensenova", {})
        return cfg.get("base_url", "https://token.sensenova.cn/v1")

    @property
    def fallback_sensenova_api_key(self) -> str:
        """SenseNova API Key（动态读取）"""
        cfg = self._config.get("fallback", {}).get("sensenova", {})
        env_names = [cfg.get("api_key_env"), "SENSENOVA_API_KEY"]
        for name in env_names:
            if not name:
                continue
            value = self._read_secret(name)
            if value:
                return value
        return ""

    # ---- 服务配置 ----

    @property
    def fastapi_host(self) -> str:
        return self._config["server"]["fastapi_host"]

    @property
    def fastapi_port(self) -> int:
        return self._config["server"]["fastapi_port"]

    # ---- RAG / 记忆配置 ----

    @property
    def rag_config(self) -> dict:
        """获取 RAG 配置（kb/chunks.json 路径、预算、top_k、tier 权重等）"""
        return self._config.get("rag", {})

    @property
    def memory_config(self) -> dict:
        """获取对话记忆配置（滑动窗口、压缩阈值等）"""
        return self._config.get("memory", {})

    def get(self, key: str, default: Any = None) -> Any:
        """通用配置获取方法"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


# 全局单例
settings = Settings()