from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from utils import logger
import httpx

LOCAL_LLM_URL = "http://localhost:9090/v1/messages"
LOCAL_LLM_KEY = "sk-1072f90d-043e-4273-8d3d-6986dc65"
LOCAL_LLM_MODEL = "deepseek-r1:14b"

class LocalLLM(LLM):
    """本地 Ollama LLM 实现"""

    client: Any = None

    def __init__(self):
        super().__init__()
        self.client = httpx.Client(
            base_url="http://localhost:9090",
            headers={"Authorization": f"Bearer {LOCAL_LLM_KEY}"},
            timeout=120.0
        )

    @property
    def _llm_type(self) -> str:
        return "local_ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            resp = self.client.post(
                "/v1/messages",
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.3),
                    "max_tokens": kwargs.get("max_tokens", 1500),
                    **({"stop": stop} if stop else {})
                }
            )
            resp.raise_for_status()
            data = resp.json()
            # 兼容 /v1/messages (Anthropic style) 和 /v1/chat/completions (OpenAI style)
            if "content" in data and data["content"]:
                return data["content"][0]["text"]
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            logger.error(f"LLM返回异常结构: {data}")
            raise ValueError(f"LLM返回空内容: {data}")
        except Exception as e:
            logger.error(f"调用本地LLM时出错: {str(e)}")
            raise

class RAGEngine:
    """RAG引擎:向量检索和回答生成"""
    def __init__(self, text_processor=None):
        try:
            self.llm = LocalLLM()
            logger.info("成功初始化本地LLM")
        except Exception as e:
            logger.error(f"初始化本地LLM失败: {str(e)}")
            raise
        self.text_processor = text_processor
        self.conversation_history: List[Dict[str, str]] = []
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            你是一个专业的助手，基于提供的参考资料回答用户问题。
            参考资料：
            {context}           
            对话历史：
            {history} 
            用户当前问题：
            {query}
            请基于参考资料和对话历史，以自然、友好的语言回答用户问题。
            若回答中包含引用来源，请必须包含引用来源，格式为"（来源：文件名）",
            若回答中不包含引用来源，请不要在回答中添加任何来源信息。
            如果参考资料中没有相关信息，请坦诚告知用户。
            """
        )
        logger.info("RAG引擎初始化完成")

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            source_info = f"来源: {metadata.get('file_name', 'unknown')}"
            context_parts.append(f"[{i+1}]")
            context_parts.append(result["text"])
            context_parts.append(f"({source_info})")
            context_parts.append("---")
        return "\n".join(context_parts)

    def format_history(self) -> str:
        if not self.conversation_history:
            return "暂无"
        history_parts = []
        for i, turn in enumerate(self.conversation_history):
            if turn["role"] == "user":
                history_parts.append(f"用户 {i//2 + 1}: {turn['content']}")
            elif turn["role"] == "assistant":
                history_parts.append(f"助手 {i//2 + 1}: {turn['content']}")
        return "\n".join(history_parts)

    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        context = self.format_context(search_results)
        history = self.format_history()
        logger.info(f"开始生成回答，查询: {query[:50]}..., 上下文长度: {len(context)}")
        try:
            chain = (
                {
                    "context": RunnablePassthrough(),
                    "history": RunnablePassthrough(),
                    "query": RunnablePassthrough()
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke({"context": context, "history": history, "query": query})
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            raise

    def rag_pipeline(self, query: str, k: int = 3) -> Dict[str, Any]:
        search_results = self.text_processor.search_similar(query, k=k)
        answer = self.generate_answer(query, search_results)
        self.add_to_history("user", query)
        self.add_to_history("assistant", answer)
        return {"query": query, "search_results": search_results, "answer": answer, "history": self.conversation_history.copy()}

    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

    def clear_history(self) -> None:
        self.conversation_history.clear()
        logger.info("对话历史已清空")

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

rag_engine = None

__all__ = ["RAGEngine", "rag_engine"]

