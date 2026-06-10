from typing import List, Dict, Any
from utils import logger


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def _load(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            logger.info(f"加载 Reranker 模型: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
            self._torch = torch
            logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return results
        try:
            self._load()
            pairs = [[query, r["text"]] for r in results]
            inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with self._torch.no_grad():
                scores = self._model(**inputs).logits.squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
            logger.info(f"Rerank 完成，top-1 分数: {ranked[0]['rerank_score']:.4f}")
            return ranked
        except Exception as e:
            logger.warning(f"Rerank 失败，返回原始顺序: {e}")
            return results[:top_k]
