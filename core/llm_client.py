"""
LLM API 调用封装模块
封装对智谱 GLM-4.7-Flash 模型的 API 调用，使用 OpenAI 兼容接口。
内置：429 限流指数退避重试、混合推理模型 thinking 开关、流式输出、
全局并发闸门（超额请求排队而非报 429）、多 Key 轮换、
模型降级链（主模型受限/超时等异常时按顺序切换备用模型）与快速切换策略。
文本降级链：glm-4.7-flash(双Key) → glm-4.6v-flash(双Key) → sensenova-6.8-flash-lite
→ OpenRouter nemotron → glm-4.6v（末位兜底）；
视觉降级链：glm-4.6v-flash(双Key) → sensenova-6.8-flash-lite → glm-4.6v。
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, RateLimitError

from core.config import settings

_MAX_RETRIES = 4              # 无降级链时的默认尝试次数（首次请求 + 3 次重试）
_FAST_CUTOVER_ATTEMPTS = 2    # 有降级链时主模型仅尝试 2 次：尽快切换而非原地久等
_RETRY_BASE_DELAY = 3.0       # 默认退避基数（秒），指数递增（3s → 6s → 12s）
_FAST_RETRY_BASE_DELAY = 1.0  # 有降级链时的退避基数（1s → 2s），把切换等待压到秒级
_FALLBACK_RETRIES = 2         # 降级链中每个备用模型的尝试次数（首次 + 1 次重试）
_REQUEST_TIMEOUT = 40.0       # 单次请求超时（秒）：超时视为"无法正常工作"，触发降级


class ModelFallbackSignal:
    """流式降级信号：首个 token 返回前发生模型切换时由 chat_stream 产出。
    消费方应据此更新等待提示（如"当前访问量过大"），不得当作正文内容展示。"""

    __slots__ = ()

    def __repr__(self):
        return "<ModelFallbackSignal>"


MODEL_FALLBACK_SIGNAL = ModelFallbackSignal()

# 后端 API 模式的降级标记：app.py 收到 MODEL_FALLBACK_SIGNAL 时改发该字符串，
# ui.py 的 API 生成器检测到后同样置位降级提示（\x01 控制符不会出现在正文里）
API_FALLBACK_MARKER = "\x01MODEL_FALLBACK\x01"

# 全局并发闸门：智谱免费档按"同时在处理的请求数"限流（并发超限直接返回 429），
# 而本系统一次问答会触发多次调用（路由分类 + 答案生成 + 记忆压缩/视觉理解），
# 多用户同时提问时这些调用彼此重叠，会"自己打满"并发额度造成自致限流。
# 闸门把超出上限的请求改为排队等待（前端照常播放思考动画），
# 用少量排队延迟换取自致 429 基本消失；上限可在 config.yaml 的 model.concurrency 调整。
_REQUEST_GATE = threading.BoundedSemaphore(max(1, int(settings.model_concurrency)))


def _safe_close(response):
    """尽力关闭未读完的流，避免连接泄漏"""
    try:
        response.close()
    except Exception:
        pass


def _make_openai_client(api_key: str, base_url: str) -> OpenAI:
    """创建 OpenAI 兼容客户端：显式超时保证降级可及时触发；
    关闭 SDK 内置重试，重试节奏统一由 _create_with_retry 控制"""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=_REQUEST_TIMEOUT, max_retries=0)


def _create_with_retry(client: OpenAI, max_attempts: int,
                       base_delay: Optional[float] = None, **kwargs):
    """带指数退避的请求封装（非流式）：免费模型高峰期频繁返回 429，
    仅对限流(429)重试；超时/连接异常直接抛出，交由降级链切换下一个模型。
    每次实际发请求前先过全局并发闸门，拿到名额才真正发出。"""
    delay = base_delay if base_delay is not None else _RETRY_BASE_DELAY
    for attempt in range(max_attempts):
        _REQUEST_GATE.acquire()
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay * (2 ** attempt))
        finally:
            _REQUEST_GATE.release()


def _attempt_plan(index: int, has_chain: bool, retries: Optional[int]):
    """计算第 index 个端点的尝试次数与退避基数：
    主端点在存在降级链时采用"少量尝试 + 秒级退避"的快速切换策略；
    显式传入 retries 的调用方（如批量评测）仍按其指定次数执行"""
    if index == 0:
        attempts = retries if retries is not None else (
            _FAST_CUTOVER_ATTEMPTS if has_chain else _MAX_RETRIES)
        base_delay = _FAST_RETRY_BASE_DELAY if has_chain else None
    else:
        attempts, base_delay = _FALLBACK_RETRIES, None
    return attempts, base_delay


@dataclass
class _ModelEndpoint:
    """降级链中的一个模型端点"""
    client: OpenAI
    model_name: str
    use_thinking: bool  # 是否附带智谱 thinking extra_body（OpenRouter 等第三方接口不兼容）
    label: str          # 端点标识，用于错误信息
    extra_body: Optional[dict] = None  # 第三方接口的自定义附加字段（use_thinking 为 False 时生效）


# 用户自定义供应商预设：base_url/model 为推荐默认值，用户可显式覆盖
CUSTOM_PROVIDER_PRESETS: dict[str, dict] = {
    "zhipu": {
        "label": "智谱GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.7-flash", "use_thinking": True,
    },
    "sensenova": {
        "label": "商汤SenseNova", "base_url": "https://token.sensenova.cn/v1",
        "model": "sensenova-6.8-flash-lite", "use_thinking": False,
        "extra_body": {"reasoning_effort": "none"},
    },
    "openrouter": {
        "label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
        "model": "", "use_thinking": False,
    },
    "openai_compat": {
        "label": "OpenAI兼容", "base_url": "", "model": "", "use_thinking": False,
    },
}


def _build_custom_endpoint(custom: Optional[dict]) -> Optional[_ModelEndpoint]:
    """按请求携带的自定义配置构造端点；信息不全或供应商未知时返回 None（走内置链）"""
    if not isinstance(custom, dict):
        return None
    preset = CUSTOM_PROVIDER_PRESETS.get((custom.get("provider") or "").strip())
    api_key = (custom.get("api_key") or "").strip()
    if preset is None or not api_key:
        return None
    base_url = ((custom.get("base_url") or "").strip() or preset["base_url"]).rstrip("/")
    model = (custom.get("model") or "").strip() or preset["model"]
    if not base_url or not model:
        return None
    return _ModelEndpoint(
        client=_make_openai_client(api_key, base_url),
        model_name=model,
        use_thinking=preset["use_thinking"],
        label=f"自定义·{preset['label']}·{model}",
        extra_body=preset.get("extra_body"),
    )


class LLMClient:
    """大模型 API 客户端，封装对智谱 GLM 系列的调用

    内置：
    - 多 Key 轮换：GLM_API_KEY/GLM_API_KEY_2 每把密钥是独立的限流配额，依次加入请求序列
    - 模型降级链：glm-4.7-flash(双Key) → glm-4.6v-flash(双Key) → sensenova-6.8-flash-lite
      → OpenRouter nemotron → glm-4.6v（末位兜底），
      前序模型限流/超时等无法正常工作时自动切换到下一个备用模型
    - 快速切换：存在降级链时主模型只做少量短退避尝试，尽快切走而非长时间卡顿
    """

    def __init__(self):
        self.model_name = settings.model_name
        # 端点1..N：主模型 glm-4.7-flash（智谱），支持多把密钥轮流使用
        primary_keys = settings.glm_api_keys
        single = len(primary_keys) == 1
        self._endpoints: list[_ModelEndpoint] = [
            _ModelEndpoint(
                client=_make_openai_client(key, settings.model_base_url),
                model_name=settings.model_name,
                use_thinking=True,
                label=settings.model_name if single else f"{settings.model_name}(key{i + 1})",
            )
            for i, key in enumerate(primary_keys)
        ]
        # 备用端点按原序追加（glm-4.6v-flash 双Key轮换 / SenseNova / OpenRouter
        # nemotron / 智谱 glm-4.6v），构建完成后统一把 SenseNova 移到链首
        sn_endpoint = None
        try:
            if settings.fallback_enabled:
                # 视觉模型 glm-4.6v-flash 纯文本调用亦可，双 Key 提供独立的限流配额
                vf_keys = settings.vision_api_keys
                vf_single = len(vf_keys) == 1
                for i, key in enumerate(vf_keys):
                    self._endpoints.append(_ModelEndpoint(
                        client=_make_openai_client(key, settings.vision_model_base_url),
                        model_name=settings.vision_model_name,
                        use_thinking=True,  # 显式关闭思考：保证在 max_tokens 内产出正文
                        label=settings.vision_model_name if vf_single
                              else f"{settings.vision_model_name}(key{i + 1})",
                    ))
                sn_key = settings.fallback_sensenova_api_key
                sn_model = settings.fallback_sensenova_model
                if sn_key and sn_model:
                    sn_endpoint = _ModelEndpoint(
                        client=_make_openai_client(sn_key, settings.fallback_sensenova_base_url),
                        model_name=sn_model,
                        use_thinking=False,  # SenseNova 网关拒绝参数表之外的字段（含 thinking）
                        extra_body={"reasoning_effort": "none"},  # 关闭默认推理：避免思考拖慢响应
                        label=sn_model,
                    )
                    self._endpoints.append(sn_endpoint)
                or_key = settings.fallback_openrouter_api_key
                or_model = settings.fallback_openrouter_model
                if or_key and or_model:
                    self._endpoints.append(_ModelEndpoint(
                        client=_make_openai_client(or_key, settings.fallback_openrouter_base_url),
                        model_name=or_model,
                        use_thinking=False,
                        label=or_model,
                    ))
                u_key = settings.ultimate_api_key
                u_model = settings.ultimate_model
                if u_key and u_model:
                    self._endpoints.append(_ModelEndpoint(
                        client=_make_openai_client(u_key, settings.ultimate_base_url),
                        model_name=u_model,
                        use_thinking=True,
                        label=u_model,
                    ))
        except Exception:
            pass  # 降级端点构建失败不影响主模型可用性

        # 主用 SenseNova 置顶：额度充足且关闭推理直出正文，首字延迟远低于高峰拥塞的
        # GLM 免费档；链首端点还享受"少量尝试+秒级退避"的快速切换策略，
        # GLM 系列整体降为备用，仅在 SenseNova 异常时才被依次尝试
        if sn_endpoint is not None:
            self._endpoints.remove(sn_endpoint)
            self._endpoints.insert(0, sn_endpoint)

    @property
    def endpoints(self) -> list[_ModelEndpoint]:
        """当前降级链端点列表（浅拷贝暴露，便于测试与诊断）"""
        return list(self._endpoints)

    def _chain(self, custom: Optional[dict] = None) -> list[_ModelEndpoint]:
        """本次调用使用的端点链：用户自定义端点（若有效）插到链首优先使用，
        失败时自动回退内置降级链；不修改全局共享的 self._endpoints"""
        custom_ep = _build_custom_endpoint(custom)
        return [custom_ep, *self._endpoints] if custom_ep is not None else self._endpoints

    @staticmethod
    def _thinking_extra_body() -> dict:
        """glm-4.7 系列为混合推理模型：思考 token 会挤占 max_tokens，
        导致非流式调用在思考阶段就被截断而返回空正文。默认关闭。"""
        return {
            "thinking": {
                "type": "enabled" if settings.model_thinking else "disabled"
            }
        }

    def _build_kwargs(
        self,
        ep: _ModelEndpoint,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        use_thinking: Optional[bool] = None,
    ) -> dict:
        kwargs = {
            "model": ep.model_name,
            "messages": messages,
            "temperature": temperature or settings.model_temperature,
            "max_tokens": max_tokens or settings.model_max_tokens,
            "top_p": top_p or settings.model_top_p,
        }
        enable_thinking = ep.use_thinking if use_thinking is None else use_thinking
        if enable_thinking:
            kwargs["extra_body"] = self._thinking_extra_body()
        elif ep.extra_body:
            kwargs["extra_body"] = ep.extra_body
        return kwargs

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        retries: Optional[int] = None,
        use_thinking: Optional[bool] = None,
        custom: Optional[dict] = None,
    ) -> str:
        """
        发送聊天请求并获取回复（主模型失败时自动降级到备用模型）

        Args:
            messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            retries: 主模型 429 最大尝试次数，默认按是否存在降级链自动决定
                     （有降级链时仅 2 次快速尝试）；备用模型固定使用 _FALLBACK_RETRIES
            use_thinking: 思考模式开关覆盖（None=沿用端点配置；False=强制关闭，
                          用于延伸问题等低延迟辅助调用）
            custom: 用户自定义模型服务配置（可选）：{"provider", "api_key",
                    "model"?, "base_url"?}；有效时插到降级链最前优先使用

        Returns:
            模型生成的回复文本
        """
        last_error = None
        endpoints = self._chain(custom)
        has_chain = len(endpoints) > 1
        for i, ep in enumerate(endpoints):
            attempts, base_delay = _attempt_plan(i, has_chain, retries)
            try:
                response = _create_with_retry(
                    ep.client, attempts, base_delay,
                    **self._build_kwargs(ep, messages, temperature, max_tokens,
                                         top_p, use_thinking),
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e  # 当前模型不可用，尝试降级链中的下一个模型
        raise last_error

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        retries: Optional[int] = None,
        custom: Optional[dict] = None,
    ):
        """
        流式聊天接口（主模型失败时自动降级到备用模型）

        并发闸门语义：建连成功后持续持有名额直到流结束，避免多个流式回答
        彼此重叠挤占智谱并发额度；后续请求在闸门前排队（前端显示思考动画）。

        降级策略：缓冲至拿到第一个内容块才对外产出——若建连/首块阶段失败
        （限流、超时、空响应等）则切换下一个模型；一旦开始输出即锁定该模型，
        此后中断不再降级（避免同一回答拼接两个模型的文本）。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            retries: 主模型 429 限流重试次数（None 则按是否存在降级链自动决定）
            custom: 用户自定义模型服务配置（可选）：{"provider", "api_key",
                    "model"?, "base_url"?}；有效时插到降级链最前优先使用

        Yields:
            若发生降级：先产出一个 MODEL_FALLBACK_SIGNAL，再产出文本片段；
            否则仅产出文本片段。
        """
        last_error = None
        endpoints = self._chain(custom)
        has_chain = len(endpoints) > 1
        for i, ep in enumerate(endpoints):
            attempts, base_delay = _attempt_plan(i, has_chain, retries)
            delay = base_delay if base_delay is not None else _RETRY_BASE_DELAY

            response = None
            gate_held = False
            streaming_started = False
            try:
                # 建连阶段：每次尝试先过闸门拿并发名额，失败立即释放再退避
                for attempt in range(attempts):
                    _REQUEST_GATE.acquire()
                    gate_held = True
                    try:
                        response = ep.client.chat.completions.create(
                            stream=True,
                            **self._build_kwargs(ep, messages, temperature, max_tokens, top_p),
                        )
                        break  # 建连成功：持有闸门直到流结束，防止其他调用挤占额度
                    except RateLimitError:
                        _REQUEST_GATE.release()
                        gate_held = False
                        if attempt == attempts - 1:
                            raise
                        time.sleep(delay * (2 ** attempt))
                if response is None:
                    last_error = RuntimeError(f"[{ep.label}] 未建立连接")
                    continue

                # 缓冲至拿到第一个内容块：此阶段失败可安全切换下一个模型
                first_content = None
                for chunk in response:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    if delta and delta.content:
                        first_content = delta.content
                        break
                if first_content is None:
                    last_error = RuntimeError(f"[{ep.label}] 未返回有效内容")
                    continue

                # 首个内容块已到手：锁定该模型开始输出，此后中断不再降级
                streaming_started = True
                if i > 0:
                    yield MODEL_FALLBACK_SIGNAL
                yield first_content
                for chunk in response:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as e:
                if streaming_started:
                    # 已开始输出后的异常向上抛给调用方，不再降级拼接文本
                    raise RuntimeError(f"[{ep.label}] 流式传输中断: {e}") from e
                last_error = e  # 首块阶段异常：记录并尝试下一个模型
                continue
            finally:
                self._safe_close_stream(response)
                if gate_held:
                    _REQUEST_GATE.release()
        raise last_error

    @staticmethod
    def _safe_close_stream(response):
        """尽力关闭未读完的流（None 安全），配合 finally 保证闸门必然释放"""
        if response is not None:
            _safe_close(response)

    # ---- 扩展预留：多轮对话历史管理 ----
    # def chat_with_history(self, messages, history=None, **kwargs):
    #     """带历史记录的对话（预留）"""
    #     full_messages = (history or []) + messages
    #     return self.chat(full_messages, **kwargs)


class VisionLLMClient:
    """视觉大模型客户端，识图请求降级链：
    glm-4.6v-flash(GLM_V_API_KEY/GLM_V_API_KEY_2 轮换)
    → sensenova-6.8-flash-lite → glm-4.6v（GLM_V_API_KEY_3，末位兜底）

    视觉请求同样受全局并发闸门约束：与文本问答共享并发额度，
    不加闸门的图片请求会在高峰期挤占/触发整体 429。
    流式降级为静默切换：现有消费方（ui.py/app.py）未处理 MODEL_FALLBACK_SIGNAL，
    故不发信号，避免被当作正文渲染。"""

    def __init__(self):
        self.model_name = settings.vision_model_name
        # 端点1..N：glm-4.6v-flash 多 Key 轮换
        v_keys = settings.vision_api_keys
        v_single = len(v_keys) == 1
        self._endpoints: list[_ModelEndpoint] = [
            _ModelEndpoint(
                client=_make_openai_client(key, settings.vision_model_base_url),
                model_name=settings.vision_model_name,
                use_thinking=False,
                label=settings.vision_model_name if v_single
                      else f"{settings.vision_model_name}(key{i + 1})",
            )
            for i, key in enumerate(v_keys)
        ]
        # 备用端点：SenseNova → 智谱 glm-4.6v（末位兜底）
        try:
            if settings.fallback_enabled:
                sn_key = settings.fallback_sensenova_api_key
                sn_model = settings.fallback_sensenova_model
                if sn_key and sn_model:
                    self._endpoints.append(_ModelEndpoint(
                        client=_make_openai_client(sn_key, settings.fallback_sensenova_base_url),
                        model_name=sn_model,
                        use_thinking=False,
                        extra_body={"reasoning_effort": "none"},  # 关闭默认推理
                        label=sn_model,
                    ))
                u_key = settings.ultimate_api_key
                u_model = settings.ultimate_model
                if u_key and u_model:
                    self._endpoints.append(_ModelEndpoint(
                        client=_make_openai_client(u_key, settings.ultimate_base_url),
                        model_name=u_model,
                        use_thinking=False,
                        label=u_model,
                    ))
        except Exception:
            pass  # 降级端点构建失败不影响主模型可用性

    @property
    def endpoints(self) -> list[_ModelEndpoint]:
        """当前降级链端点列表（浅拷贝暴露，便于测试与诊断）"""
        return list(self._endpoints)

    @staticmethod
    def _build_kwargs(
        ep: _ModelEndpoint,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        kwargs = {
            "model": ep.model_name,
            "messages": messages,
            "temperature": temperature or settings.vision_model_temperature,
            "max_tokens": max_tokens or settings.vision_model_max_tokens,
            "top_p": top_p or settings.vision_model_top_p,
        }
        if ep.use_thinking:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if settings.model_thinking else "disabled"}
            }
        elif ep.extra_body:
            kwargs["extra_body"] = ep.extra_body
        return kwargs

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """发送多模态聊天请求并获取回复（失败时沿降级链自动切换）"""
        last_error = None
        has_chain = len(self._endpoints) > 1
        for i, ep in enumerate(self._endpoints):
            attempts, base_delay = _attempt_plan(i, has_chain, None)
            try:
                response = _create_with_retry(
                    ep.client, attempts, base_delay,
                    **self._build_kwargs(ep, messages, temperature, max_tokens, top_p),
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e  # 当前模型不可用，尝试下一个模型
        raise last_error

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """流式多模态聊天接口（失败时沿降级链静默切换）

        与文本链相同的缓冲-锁定语义：缓冲至拿到第一个内容块才对外产出，
        首块前失败可安全换模型；开始输出后中断则抛错（不拼接两个模型的文本）。"""
        last_error = None
        has_chain = len(self._endpoints) > 1
        for i, ep in enumerate(self._endpoints):
            attempts, base_delay = _attempt_plan(i, has_chain, None)
            delay = base_delay if base_delay is not None else _RETRY_BASE_DELAY

            response = None
            gate_held = False
            streaming_started = False
            try:
                # 建连阶段：每次尝试先过闸门拿并发名额，失败立即释放再退避
                for attempt in range(attempts):
                    _REQUEST_GATE.acquire()
                    gate_held = True
                    try:
                        response = ep.client.chat.completions.create(
                            stream=True,
                            **self._build_kwargs(ep, messages, temperature, max_tokens, top_p),
                        )
                        break  # 建连成功：持有闸门直到流结束
                    except RateLimitError:
                        _REQUEST_GATE.release()
                        gate_held = False
                        if attempt == attempts - 1:
                            raise
                        time.sleep(delay * (2 ** attempt))
                if response is None:
                    last_error = RuntimeError(f"[{ep.label}] 未建立连接")
                    continue

                # 缓冲至拿到第一个内容块：此阶段失败可安全切换下一个模型
                first_content = None
                for chunk in response:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    if delta and delta.content:
                        first_content = delta.content
                        break
                if first_content is None:
                    last_error = RuntimeError(f"[{ep.label}] 未返回有效内容")
                    continue

                streaming_started = True
                yield first_content
                for chunk in response:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as e:
                if streaming_started:
                    raise RuntimeError(f"[{ep.label}] 流式传输中断: {e}") from e
                last_error = e
                continue
            finally:
                LLMClient._safe_close_stream(response)
                if gate_held:
                    _REQUEST_GATE.release()
        raise last_error


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
