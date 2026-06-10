import os
import json
from dotenv import load_dotenv
from utils import logger, configure_standard_logging, ensure_directory
from document_processor import document_processor
from text_processor import text_processor
from rag_engine import RAGEngine
from agent_flow import RAGAgent
import sys

def setup_environment():
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("环境变量加载成功")
    else:
        logger.warning(".env文件不存在，请根据.env.example创建")
    
    configure_standard_logging()
    
    ensure_directory("data")
    ensure_directory("vector_db")
    ensure_directory("logs")

def create_rag_system():
    rag_engine = RAGEngine(text_processor)
    agent = RAGAgent(
        document_processor=document_processor,
        text_processor=text_processor,
        rag_engine=rag_engine
    )
    return agent

def main():
    setup_environment()
    logger.info("=== RAG系统启动 ===")
    agent = create_rag_system()
    return agent

if __name__ == "__main__":
    import uvicorn
    setup_environment()
    logger.info("=== RAG Web服务启动 === http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

__all__ = ["setup_environment", "create_rag_system", "main"]
