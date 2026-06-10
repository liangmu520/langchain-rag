"""
RAG 评估：基于 RAGAS 框架，评估 faithfulness / answer_relevancy / context_recall。
简历亮点：RAG Evaluation Pipeline (RAGAS)
"""
from typing import List, Dict, Any
from utils import logger


class RAGEvaluator:
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """
        返回评估指标字典。
        ground_truth 可选，提供时额外计算 context_recall。
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy

            data: Dict[str, List] = {
                "question":  [question],
                "answer":    [answer],
                "contexts":  [contexts],
            }
            metrics = [faithfulness, answer_relevancy]

            if ground_truth:
                from ragas.metrics import context_recall
                data["ground_truth"] = [ground_truth]
                metrics.append(context_recall)

            dataset = Dataset.from_dict(data)
            result  = evaluate(dataset, metrics=metrics)
            scores  = {k: round(float(v), 4) for k, v in result.items()}
            logger.info(f"RAGAS 评估结果: {scores}")
            return scores
        except Exception as e:
            logger.warning(f"RAGAS 评估失败: {e}")
            return {"error": str(e)}

    def batch_evaluate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量评估，samples 每项含 question/answer/contexts[/ground_truth]"""
        return [
            {**s, "ragas": self.evaluate(
                s["question"], s["answer"], s["contexts"], s.get("ground_truth", "")
            )}
            for s in samples
        ]


evaluator = RAGEvaluator()
__all__ = ["RAGEvaluator", "evaluator"]
