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

        # 读取视觉模型 API Key（可选，未配置时禁用视觉功能）
        self.GLM_V_API_KEY = self._read_secret("GLM_V_API_KEY")

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

    # ---- 服务配置 ----

    @property
    def fastapi_host(self) -> str:
        return self._config["server"]["fastapi_host"]

    @property
    def fastapi_port(self) -> int:
        return self._config["server"]["fastapi_port"]

    # ---- 扩展预留接口 ----

    # def get_rag_config(self) -> dict:
    #     """获取 RAG 配置（预留）"""
    #     return self._config.get("rag", {})

    # def get_theme_config(self) -> dict:
    #     """获取主题配置（预留）"""
    #     return self._config.get("theme", {})

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