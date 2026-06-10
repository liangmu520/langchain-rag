"""
混合检索：FAISS 向量检索 + BM25 关键词检索，RRF 融合排序。
简历亮点：Hybrid Search + Reciprocal Rank Fusion
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from utils import logger


def _rrf(rankings: List[List[int]], k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion"""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self._corpus: List[str] = []
        self._meta: List[Dict] = []
        self._bm25: BM25Okapi = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        """从 chunks 构建 BM25 索引（每次入库新文档后调用）"""
        self._corpus = [c["text"] for c in chunks]
        self._meta = [{"text": c["text"], "metadata": {k: v for k, v in c.items() if k != "text"}} for c in chunks]
        tokenized = [list(t) for t in self._corpus]   # 字符级分词，兼容中文
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引构建完成，文档数: {len(self._corpus)}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._bm25 or not self.text_processor.vector_store:
            logger.warning("索引未就绪，降级为纯向量检索")
            return self.text_processor.search_similar(query, k=k)

        # --- 向量检索 top-k*2 ---
        vec_results = self.text_processor.search_similar(query, k=k * 2)
        vec_texts = [r["text"] for r in vec_results]

        # --- BM25 检索 top-k*2 ---
        bm25_scores = self._bm25.get_scores(list(query))
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: k * 2]
        bm25_texts = [self._corpus[i] for i in bm25_top_idx]

        # --- 统一文档池 ---
        all_texts = list(dict.fromkeys(vec_texts + bm25_texts))   # 去重保序

        vec_rank   = [all_texts.index(t) for t in vec_texts  if t in all_texts]
        bm25_rank  = [all_texts.index(t) for t in bm25_texts if t in all_texts]

        fused = _rrf([vec_rank, bm25_rank])[:k]

        # --- 组装结果 ---
        text_to_vec = {r["text"]: r for r in vec_results}
        results = []
        for idx, rrf_score in fused:
            text = all_texts[idx]
            if text in text_to_vec:
                entry = dict(text_to_vec[text])
                entry["rrf_score"] = round(rrf_score, 6)
                entry["retrieval"] = "hybrid"
            else:
                bm25_pos = bm25_texts.index(text)
                orig_idx = bm25_top_idx[bm25_pos]
                meta = self._meta[orig_idx] if orig_idx < len(self._meta) else {}
                entry = {
                    "text": text,
                    "score": float(bm25_scores[orig_idx]),
                    "metadata": meta.get("metadata", {}),
                    "rrf_score": round(rrf_score, 6),
                    "retrieval": "bm25"
                }
            results.append(entry)

        logger.info(f"混合检索完成，返回 {len(results)} 条，来源: {[r['retrieval'] for r in results]}")
        return results
