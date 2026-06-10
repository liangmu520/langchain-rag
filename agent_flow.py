from typing import Any, List, Optional
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils import logger
import os
import time
from langsmith.client import Client

from hybrid_retriever import HybridRetriever
from query_rewriter import QueryRewriter
from reranker import Reranker


class AgentState(TypedDict, total=False):
    query: str
    search_results: List[Dict[str, Any]]
    answer: str
    history: List[Dict[str, str]]
    intermediate_steps: List[Dict[str, Any]]
    start_time: float
    metrics: Dict[str, Any]


class RAGAgent:
    def __init__(self, document_processor, text_processor, rag_engine):
        self.document_processor = document_processor
        self.text_processor = text_processor
        self.rag_engine = rag_engine
        self.rag_engine.text_processor = text_processor

        # ── 高级检索组件 ───────────────────────────────────────────────────────
        self.hybrid_retriever = HybridRetriever(text_processor)
        self.reranker = Reranker()                       # 懒加载模型
        self.query_rewriter = QueryRewriter(
            llm_call=self._llm_call,
            hybrid_retriever=self.hybrid_retriever,
        )
        self._all_chunks: List[Dict[str, Any]] = []     # 累积 chunks，供 BM25 重建

        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            self.langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            logger.info("LangSmith客户端初始化成功")
        else:
            self.langsmith_client = None

        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())
        logger.info("RAG Agent初始化完成（Hybrid+HyDE+Rerank）")

    # ── LLM 裸调用（供 QueryRewriter 使用）────────────────────────────────────
    def _llm_call(self, prompt: str) -> str:
        return self.rag_engine.llm._call(prompt)

    # ── LangGraph ─────────────────────────────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve",   self._retrieve_documents)
        graph.add_node("rerank",     self._rerank_documents)
        graph.add_node("generate",   self._generate_answer)
        graph.add_node("log_result", self._log_result)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve",   "rerank")
        graph.add_edge("rerank",     "generate")
        graph.add_edge("generate",   "log_result")
        graph.add_edge("log_result", END)
        logger.info("LangGraph 流程构建完成：retrieve→rerank→generate→log")
        return graph

    def _retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        logger.info(f"[retrieve] 查询: {query[:50]}...")

        if not self.text_processor.vector_store:
            return {**state, "search_results": [], "answer": "请先上传文档，再进行提问。"}

        # 有 BM25 索引 → 高级检索；否则降级
        if self.hybrid_retriever._bm25:
            results = self.query_rewriter.search(query, k=8)
        else:
            results = self.text_processor.search_similar(query, k=8)

        steps = state.get("intermediate_steps", [])
        steps.append({"node": "retrieve", "count": len(results), "timestamp": time.time()})
        return {**state, "search_results": results, "intermediate_steps": steps}

    def _rerank_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query   = state.get("query", "")
        results = state.get("search_results", [])
        logger.info(f"[rerank] 候选文档数: {len(results)}")
        reranked = self.reranker.rerank(query, results, top_k=5)
        steps = state.get("intermediate_steps", [])
        steps.append({"node": "rerank", "count": len(reranked), "timestamp": time.time()})
        return {**state, "search_results": reranked, "intermediate_steps": steps}

    def _generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query   = state.get("query", "")
        results = state.get("search_results", [])
        logger.info("[generate] 开始生成回答")
        answer = self.rag_engine.generate_answer(query, results)
        self.rag_engine.add_to_history("user", query)
        self.rag_engine.add_to_history("assistant", answer)
        steps = state.get("intermediate_steps", [])
        steps.append({"node": "generate", "answer": answer, "timestamp": time.time()})
        return {**state, "answer": answer, "history": self.rag_engine.get_history(), "intermediate_steps": steps}

    def _log_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query   = state.get("query", "")
        answer  = state.get("answer", "")
        results = state.get("search_results", [])
        elapsed = time.time() - state.get("start_time", time.time())
        logger.info(f"[log_result] 响应时间: {elapsed:.2f}s")

        if self.langsmith_client:
            try:
                scores = [r.get("rerank_score", r.get("score", 0)) for r in results]
                self.langsmith_client.create_run(
                    name="rag_query", run_type="chain",
                    inputs={"query": query, "results_count": len(results)},
                    outputs={"answer": answer, "response_time": elapsed,
                             "avg_score": sum(scores) / len(scores) if scores else 0},
                    tags=["rag", "hybrid", "rerank"],
                )
            except Exception as e:
                logger.warning(f"LangSmith 记录失败: {e}")

        return {**state, "metrics": {"response_time": elapsed,
                                     "rerank_scores": [r.get("rerank_score") for r in results]}}

    # ── 公开接口 ──────────────────────────────────────────────────────────────
    def run(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Agent 开始处理: {query[:50]}...")
        initial_state = {
            "query": query,
            "search_results": [],
            "answer": "",
            "history": self.rag_engine.get_history(),
            "intermediate_steps": [],
            "start_time": time.time(),
        }
        return self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "rag_thread_01"}},
        )

    def process_document(self, file_path: str) -> Dict[str, Any]:
        document = self.document_processor.process_file(file_path)
        chunks   = self.text_processor.process_document(document)
        # 累积 chunks，重建 BM25 索引
        self._all_chunks.extend(chunks)
        self.hybrid_retriever.build_index(self._all_chunks)
        return {"document": document, "chunks_count": len(chunks)}


__all__ = ["RAGAgent", "AgentState"]

