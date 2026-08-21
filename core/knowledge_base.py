"""
本地知识库检索模块（Step 1）
基于 jieba 分词 + BM25 的轻量级检索，无网络开销、毫秒级返回。
检索流程：BM25 初筛 top_k → tier 加权重排 → 返回 final_k 个块。

设计要点：
- 模块级单例，Streamlit 与 FastAPI 两种运行环境均可安全复用
- tier 权重使自有文档（一稿/选题报告/案例文本）在分数相近时优先于文献汇编
"""

import json
import threading
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from core.config import settings

ROOT = Path(__file__).resolve().parent.parent

# 领域词典：保证"交易成本/报表通/台账授权"等术语在查询与文档两侧分词一致
_userdict = ROOT / "kb" / "user_dict.txt"
if _userdict.exists():
    jieba.load_userdict(str(_userdict))

# 同义词扩展表：查询命中 key 时追加扩展词参与 BM25 匹配，弥合词汇鸿沟
_synonyms_path = ROOT / "kb" / "synonyms.json"
SYNONYMS: dict[str, list[str]] = {}
if _synonyms_path.exists():
    with open(_synonyms_path, "r", encoding="utf-8") as f:
        SYNONYMS = json.load(f)


class KnowledgeBase:
    """本地 BM25 知识库"""

    def __init__(self, chunks_path: Path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)
        # 预分词建索引
        self._tokenized = [list(jieba.cut(c["content"])) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        tier_filter: list[int] | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        检索与 query 最相关的知识块

        Args:
            query: 用户问题
            top_k: BM25 初筛数量
            final_k: 重排后返回数量
            tier_filter: 仅检索指定 tier（None 表示全部）
            min_score: 加权后分数低于该值的块不返回（结果为空时兜底返回前 2 条）

        Returns:
            按相关性排序的知识块列表（含 score 字段）
        """
        cfg = settings.rag_config
        top_k = top_k or cfg.get("bm25_top_k", 20)
        final_k = final_k or cfg.get("final_top_k", 6)
        min_score = min_score if min_score is not None else cfg.get("min_score", 0.8)
        tier_weights = cfg.get("tier_weights", {"1": 1.35, "2": 1.0})

        query_tokens = list(jieba.cut(query))
        # 同义词扩展：查询包含同义词表 key 时，追加扩展词提升召回
        for key, expansions in SYNONYMS.items():
            if key in query:
                query_tokens.extend(expansions)
        with self._lock:
            scores = self._bm25.get_scores(query_tokens)

        candidates = []
        # 长度>=2 的查询词用于章节标题匹配（单字无区分度）
        title_tokens = [t for t in query_tokens if len(t.strip()) >= 2]
        section_bonus = float(cfg.get("section_match_bonus", 0.35))
        for i, score in enumerate(scores):
            chunk = self.chunks[i]
            if tier_filter is not None and chunk["tier"] not in tier_filter:
                continue
            weight = float(tier_weights.get(str(chunk["tier"]), 1.0))
            # 章节标题命中查询词 → 该块是主题专述块，加分提权
            bonus = (
                section_bonus
                if title_tokens
                and any(t in chunk.get("section", "") for t in title_tokens)
                else 0.0
            )
            candidates.append((score * weight * (1 + bonus), score, i))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for weighted, raw, i in candidates[:final_k]:
            if weighted < min_score and results:
                break
            chunk = dict(self.chunks[i])
            chunk["score"] = round(raw, 3)
            chunk["weighted_score"] = round(weighted, 3)
            results.append(chunk)
        # 兜底：一个都没过阈值时返回前 2 条，保证有上下文可用
        if not results and candidates:
            for weighted, raw, i in candidates[:2]:
                chunk = dict(self.chunks[i])
                chunk["score"] = round(raw, 3)
                chunk["weighted_score"] = round(weighted, 3)
                results.append(chunk)
        return results


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_kb() -> KnowledgeBase:
    """获取知识库单例（首次调用时构建，约 1-2 秒）"""
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                chunks_path = ROOT / settings.rag_config.get(
                    "chunks_path", "kb/chunks.json"
                )
                _kb = KnowledgeBase(chunks_path)
    return _kb
