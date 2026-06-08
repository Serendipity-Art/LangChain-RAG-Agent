
<p align="center">
  <img src="https://img.shields.io/badge/LangChain-1.3.4-339933?logo=langchain" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Gradio-6.17.3-F97316?logo=gradio" alt="Gradio"/>
  <img src="https://img.shields.io/badge/FAISS-1.14.2-0052CC" alt="FAISS"/>
  <img src="https://img.shields.io/badge/DeepSeek-API-4F46E5" alt="DeepSeek"/>
  <img src="https://img.shields.io/badge/Deploy-HF%20Spaces-FF9D00" alt="HF Spaces"/>
</p>

<h1 align="center">🦜 LangChain RAG Agent</h1>

<p align="center">
  <strong>可视化检索增强生成（RAG）系统</strong><br>
  上传文档 → 自动建库 → 基于知识库的智能问答
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/serendipityArt/SuZhou_yuanlian_RAG" target="_blank">
    🚀 在线体验（HuggingFace Spaces）
  </a>
</p>

---

## 📋 项目简介

**LangChain RAG Agent** 是一个开箱即用的检索增强生成（Retrieval-Augmented Generation）系统。用户只需上传 Excel 或 CSV 格式的文档语料，系统自动完成文本分割、向量化建库和问答检索的全流程。

- **通用性强**：不限领域、不限语种，任何 Excel/CSV 知识库均可接入
- **即开即用**：无需配置服务器，HuggingFace Spaces 一键部署
- **双模式问答**：标准 RAG 模式 + Agent 智能体模式
- **完全可视化**：Gradio 界面，滑块/下拉/文件上传，零代码操作

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 📂 **上传语料** | 支持 `.xlsx` / `.xls` / `.csv` 格式 |
| 🔧 **可调分割** | Chunk Size / Overlap 即时调整 |
| 🧠 **多种 Embedding** | all-MiniLM-L6-v2 / BGE 中文等 4 种可选 |
| 🔍 **标准 RAG** | 检索 → 拼接 → LLM 回答 |
| 🤖 **Agent 模式** | LangChain `create_agent` ReAct 智能体 |
| 📊 **性能统计** | 每次回答显示检索耗时 / LLM 推理耗时 |
| 📄 **检索溯源** | 可展开查看每段检索原文片段 |
| 🔑 **自带 API Key** | 每人填入自己的 Key，不共享 |
| 🌐 **多模型兼容** | DeepSeek / GPT-4o / 任何 OpenAI 兼容 API |

---

## 🏗️ 系统架构

```
用户上传 (Excel/CSV)
       ↓
📄 Document 加载
       ↓
✂️ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
       ↓
🧠 HuggingFace Embeddings (384维向量)
       ↓
📇 FAISS 向量索引 (近似检索)
       ↓
┌─────────────────────────────┐
│     两种问答模式             │
│                             │
│  ① 标准 RAG                 │
│     Retriever(k=3) →        │
│     PromptTemplate → LLM    │
│                             │
│  ② Agent 模式               │
│     create_agent + Tool     │
│     ReAct 循环推理           │
└─────────────────────────────┘
```

---

## 🚀 快速开始

### 方式一：HuggingFace Spaces（推荐）

访问在线地址，直接使用：

<a href="https://huggingface.co/spaces/serendipityArt/SuZhou_yuanlian_RAG" target="_blank">
  <img src="https://img.shields.io/badge/🤗%20Spaces-LangChain%20RAG%20Agent-FF9D00" alt="HF Spaces">
</a>

**使用步骤：**
1. 在左侧「API 设置」中填入你的 API Key 和 Base URL
2. 在「数据与向量库」上传 `.xlsx` 或 `.csv` 文件
3. 点击「🚀 构建向量库」
4. 切换到「💬 问答」选项卡，开始提问！

### 方式二：本地运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/langchain-rag-agent.git
cd langchain-rag-agent

# 安装依赖
pip install -r huggingface_spaces/requirements.txt

# 启动
python huggingface_spaces/app.py

# 浏览器打开 http://localhost:7860
```

---

## 🛠️ 技术栈

| 层级 | 技术选型 | 版本 |
|------|---------|------|
| 前端框架 | Gradio | 6.17.3 |
| LLM 框架 | LangChain | 1.3.4 |
| LLM 接口 | LangChain-OpenAI | 1.2.2 |
| 向量库 | FAISS (cpu) | 1.14.2 |
| Embedding | sentence-transformers | 5.5.1 |
| 文档处理 | pandas / openpyxl | — |
| 部署平台 | HuggingFace Spaces | — |

---

## 📈 优化路线图

### Phase 1：体验优化（近期）
- [ ] **流式输出** — LLM 生成内容逐字显示，感知延迟从 3s → 0.3s
- [ ] **多轮对话记忆** — 保留对话历史，上下文连贯问答
- [ ] **持久化 FAISS 索引** — 重启不重建，秒级加载

### Phase 2：性能提升（中期）
- [ ] **Redis 缓存** — 缓存频繁查询结果 + 对话记忆，响应速度提升 10x
- [ ] **混合检索** — 向量检索 + BM25 关键词融合，召回率提升 30%
- [ ] **上下文压缩** — LLMChainExtractor 压缩检索片段，减少 Token 消耗

### Phase 3：架构升级（远期）
- [ ] **多用户隔离** — 用户级数据隔离 + Session 管理
- [ ] **Milvus/Qdrant 分布式向量库** — 百万级以上语料
- [ ] **文档格式扩展** — PDF / Word / Markdown / 网页抓取

---

## 🧪 后续优化详解：Redis 引入

当前系统每次问答独立进行，缺乏记忆和缓存能力。引入 Redis 可解决三个核心问题：

### 1. 对话记忆（Conversation Memory）
```python
# 当前：每次问答无上下文
# 优化后：Redis 存储对话历史
redis_client.lpush(f"session:{session_id}", 
                   f"user: {query}", f"assistant: {answer}")
```
- 多轮对话上下文连贯
- 支持会话超时自动清理

### 2. 查询缓存（Query Cache）
```python
# 当前：相同问题每次都重新检索+LLM
# 优化后：Redis 缓存热门问答
cached = redis_client.get(f"q:{hash(query)}")
if cached:
    return cached  # 毫秒级返回
```
- 高频问题直接命中
- 减少 API 调用成本 40-60%

### 3. 向量缓存（Embedding Cache）
- 相同文本的 Embedding 向量缓存到 Redis
- 避免重复计算，建库速度提升 50%

---

## 📄 项目文件结构

```
langchain-rag-agent/
├── huggingface_spaces/          # HF Spaces 部署目录
│   ├── app.py                   # 主程序（Gradio 界面）
│   ├── requirements.txt         # 依赖列表
│   └── README.md                # Spaces 配置说明
├── .gitignore
└── README.md                    # 项目总说明（本文件）
```

---

## 📜 许可证

MIT License

---

<p align="center">
  <strong>从 Notebook 到产品，只差一个 Readme。</strong><br>
  <sub>Built with ❤️ by serendipityArt</sub>
</p>
