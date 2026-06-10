# RAG System

## 项目介绍

基于 LangChain / LangGraph 构建的生产级 RAG（检索增强生成）系统，提供文档上传解析、混合检索、神经网络精排、查询改写、多轮对话及 REST API 服务。

## 核心功能

| 模块 | 说明 |
|------|------|
| 文档解析 | 支持 PDF、Word (.docx)、Markdown、TXT |
| 文本分段 | `RecursiveCharacterTextSplitter` 智能分段，可配置 chunk size / overlap |
| 向量化 | 本地 BGE Embeddings（Ollama）+ FAISS 索引 |
| 混合检索 | FAISS 向量检索 + BM25 关键词检索，RRF 融合排序 |
| 查询改写 | HyDE（假设文档嵌入）+ Multi-Query 多角度扩展 |
| 精排 | `BAAI/bge-reranker-base` Cross-Encoder（transformers 原生推理） |
| 回答生成 | Qwen3（兼容 OpenAI `/v1/messages` 接口） |
| Agent 流程 | LangGraph `StateGraph`：retrieve → rerank → generate → log |
| 多轮对话 | `MemorySaver` 保存会话上下文 |
| 可观测性 | LangSmith 全链路追踪 + loguru 结构化日志 |
| REST API | FastAPI，支持文件上传、查询、会话管理 |

## 技术栈

- **Python 3.11+**
- **LangChain** — RAG 核心框架（文本分段、向量存储、LLM 抽象）
- **LangGraph** — Agent 执行流程图（StateGraph + MemorySaver）
- **LangSmith** — 可观测性与追踪
- **FAISS** — 高效向量索引
- **BM25 (rank-bm25)** — 关键词检索
- **BGE Embeddings** — 本地文本嵌入（via Ollama）
- **BAAI/bge-reranker-base** — Cross-Encoder 精排（transformers）
- **Qwen3** — 大语言模型（OpenAI 兼容 API）
- **FastAPI** — REST API 服务
- **loguru** — 结构化日志
- **uv** — 依赖管理

## 项目结构

```
RAG_System/
├── app.py                  # FastAPI REST API 服务
├── main.py                 # 命令行入口
├── demo.py                 # 演示脚本
├── agent_flow.py           # LangGraph Agent 流程
├── rag_engine.py           # RAG 引擎（prompt 构造 + LLM 调用）
├── text_processor.py       # 文本分段与向量化（FAISS）
├── hybrid_retriever.py     # 混合检索（FAISS + BM25 + RRF）
├── query_rewriter.py       # 查询改写（HyDE + Multi-Query）
├── reranker.py             # Cross-Encoder 精排
├── document_processor.py   # 文档解析（PDF / Word / MD / TXT）
├── utils.py                # 日志配置、通用工具
├── pyproject.toml          # 项目配置与依赖
└── .env                    # 环境变量（API Key 等）
```

## 安装与配置

### 1. 安装依赖

```bash
pip install uv
uv sync
```

### 2. 配置环境变量

编辑 `.env`：

```
Qwen_API_KEY=your_qwen_api_key
LangSmith_API_KEY=your_langsmith_api_key
```

### 3. 启动 Ollama（本地 Embeddings）

```bash
ollama pull bge-m3
ollama serve
```

## 启动服务

```bash
# API 服务
python app.py
# 或
start_server.bat

# 命令行演示
python demo.py
python main.py "path/to/doc.pdf" "your question"
```

## RAG 流程

```
用户提问
  → Query 改写（HyDE / Multi-Query）
  → 混合检索（FAISS 向量 + BM25 关键词 → RRF 融合）
  → Cross-Encoder 精排（bge-reranker-base）
  → Prompt 构造（上下文 + 对话历史 + 问题）
  → Qwen3 生成回答（含来源引用）
  → LangSmith 追踪记录
```
