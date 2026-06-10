"""
Query 改写：
  1. HyDE (Hypothetical Document Embeddings) — 让 LLM 生成假设答案再检索
  2. Multi-Query — 生成多角度查询，合并去重结果
简历亮点：Advanced RAG - Query Transformation
"""
from typing import List, Dict, Any
from utils import logger


class QueryRewriter:
    def __init__(self, llm_call, hybrid_retriever):
        """
        llm_call: 接受 prompt(str) 返回 str 的可调用对象
        hybrid_retriever: HybridRetriever 实例
        """
        self.llm_call = llm_call
        self.retriever = hybrid_retriever

    # ── HyDE ──────────────────────────────────────────────────────────────────
    def hyde_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """用假设文档替代原始查询进行向量检索"""
        try:
            hypo_prompt = (
                f"请用 2-3 句话写一段能回答以下问题的假设性文档片段，只输出片段内容：\n{query}"
            )
            hypo_doc = self.llm_call(hypo_prompt)
            logger.info(f"HyDE 生成假设文档: {hypo_doc[:80]}...")
            return self.retriever.search(hypo_doc, k=k)
        except Exception as e:
            logger.warning(f"HyDE 失败，降级原始查询: {e}")
            return self.retriever.search(query, k=k)

    # ── Multi-Query ───────────────────────────────────────────────────────────
    def multi_query_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """生成 3 个改写查询，合并检索结果后去重"""
        try:
            rewrite_prompt = (
                f"将以下问题改写为 3 个不同角度的搜索查询，每行一个，只输出查询本身：\n{query}"
            )
            raw = self.llm_call(rewrite_prompt)
            queries = [q.strip() for q in raw.strip().splitlines() if q.strip()][:3]
            if not queries:
                queries = [query]
            logger.info(f"Multi-Query 生成: {queries}")
        except Exception as e:
            logger.warning(f"Multi-Query 改写失败: {e}")
            queries = [query]

        seen, merged = set(), []
        for q in [query] + queries:
            for r in self.retriever.search(q, k=k):
                if r["text"] not in seen:
                    seen.add(r["text"])
                    merged.append(r)
        # 按 rrf_score 或 score 排序，取 top-k
        merged.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)
        return merged[:k]

    # ── 统一入口：HyDE + Multi-Query 融合 ─────────────────────────────────────
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        hyde_res  = self.hyde_search(query, k=k)
        multi_res = self.multi_query_search(query, k=k)

        seen, merged = set(), []
        for r in hyde_res + multi_res:
            if r["text"] not in seen:
                seen.add(r["text"])
                merged.append(r)
        merged.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)
        logger.info(f"Query改写融合完成，返回 {min(k, len(merged))} 条")
        return merged[:k]
