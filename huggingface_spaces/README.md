---
title: LangChain RAG Agent
emoji: 🦜
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
---

# LangChain RAG Agent — 可视化问答系统

基于 LangChain + FAISS + DeepSeek 的检索增强生成（RAG）系统，提供 Gradio 可视化界面。

## 功能

| 功能 | 说明 |
|------|------|
| 📂 上传语料 | 支持 `.xlsx` / `.xls` / `.csv` 格式 |
| 🔧 可配置分割 | Chunk Size / Overlap 可调 |
| 🧠 多种 Embedding | 4 种可选，含 BGE 中文优化模型，带详细说明 |
| 🏭 国内大模型厂商 | 下拉选择 DeepSeek / 智谱 / 通义千问 / Moonshot 等 7 家，自动匹配 Base URL 和模型列表 |
| 🔍 标准 RAG | 检索 → 拼接 → LLM 回答 |
| 🤖 Agent 模式 | LangChain `create_agent` 带检索工具的 ReAct Agent |
| 📊 性能统计 | 每次回答显示检索耗时 / LLM 推理耗时 / 总耗时 |
| 📄 检索溯源 | 可展开查看检索到的文档片段 |

## 在 HF Spaces 部署

### 方式一：一键部署

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-blue)](https://huggingface.co/new-space?template=https://github.com/YOUR_USER/YOUR_REPO)

1. 将本目录推送到你的 GitHub 仓库
2. 在 [HuggingFace Spaces](https://huggingface.co/new-space) 新建 Space
3. 选择 "Import from GitHub" 填入仓库地址
4. SDK 选择 **Gradio**
5. 在 Space 的 **Settings → Repository Secrets** 中添加（可选）：

| Secret 名称 | 说明 |
|------------|------|
| `OPENAI_API_KEY1` | API Key（如没填写，UI 中手动输入） |
| `OPENAI_BASE_URL1` | API Base URL（如没填写，UI 中手动输入） |

### 方式二：直接上传

将 `app.py` + `requirements.txt` 打包，在 Spaces 页面直接 Upload。

## 本地运行

```bash
cd huggingface_spaces
pip install -r requirements.txt
python app.py
```

浏览器打开 http://localhost:7860

---

# 📐 语料变大后的速度优化方案

## 当前架构瓶颈分析

你的 notebook 当前是一条线性流水线：

```
[Excel 语料] → TextSplitter → [1025 chunks] → FAISS Flat Index → Retriever(k=3) → Prompt 拼接 → LLM 回答
```

每个环节在语料量增长时的表现：

### 1️⃣ Embedding 生成（索引阶段）

| 语料规模 | `all-MiniLM-L6-v2` 耗时（估算） |
|----------|-------------------------------|
| 1k 片段 | ~2 秒 |
| 10k 片段 | ~20 秒 |
| 100k 片段 | ~3 分钟 |
| 1M 片段 | ~30 分钟 |

**现状**：每次启动重建索引，只适合 Demo。

### 2️⃣ FAISS 检索（查询阶段）

| 索引类型 | 1k 向量 | 10k | 100k | 1M |
|---------|---------|-----|------|-----|
| **Flat (当前)** | <1ms | 2ms | 15ms | 150ms |
| **IVF100, SQ8** | <1ms | <1ms | 2ms | 15ms |

**现状**：Flat 索引查询 1M 向量仍在 150ms 内，**查询速度不是当前瓶颈**。

### 3️⃣ LLM 调用（真正的瓶颈）

DeepSeek API 调用耗时约 **1-3 秒**，占单次回答总时间的 **80-95%**。语料增多主要影响 **检索到的文本量** → Prompt 变长 → LLM 推理更慢。

| 检索 Top-K | 约 tokens | LLM 耗时（估算） |
|-----------|----------|----------------|
| 3 (当前) | ~1500 | 1-2s |
| 5 | ~2500 | 2-3s |
| 10 | ~5000 | 3-5s |

---

## 优化路线图

### 🔵 第一层：零成本优化（不改架构，调参即可）

| 优化项 | 效果 | 操作 |
|--------|------|------|
| **流式输出** | 减少感知延迟（TTFP），首字 0.3s | `llm.stream()` + Gradio 流式回调 |
| **调整 chunk_size** | 减少检索冗余 | 500→800，减少片段数但保留语义完整性 |
| **启用近似检索** | 1M 向量下 150ms→15ms | `FAISS.IndexIVFFlat` 替代 `IndexFlatL2` |
| **缓存频繁查询** | 热门问题直接命中 | LRU Cache（`@functools.lru_cache` 或 Redis） |

### 🟡 第二层：架构优化（需要改部分代码）

#### 优化 1：持久化向量索引

```python
# 保存到磁盘
vs.save_local("faiss_index")

# 热加载
vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

**效果**：重启不重建，百万级索引秒级加载。

#### 优化 2：混合检索 → 重排序

```
query → [向量检索 (top-20)] → [BM25 关键词检索 (top-20)] → [交叉编码器重排序 → top-3]
```

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 向量检索 + 关键词检索 融合
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
```

**效果**：召回率提升 15-30%，减少 LLM 漏答案。

#### 优化 3：多阶段检索（Auto-Retrieval）

```
query → LLM 生成候选关键字（HyDE）→ 向量检索 → 分类器筛选 → LLM 回答
```

或使用 **Self-RAG**：让 LLM 自己判断是否需要检索。

#### 优化 4：上下文压缩

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

**效果**：每个检索片段压缩 60-80% 长度，LLM 推理速度翻倍。

### 🔴 第三层：构架级优化（大改动）

#### 方案 A：分片 + 分层索引

```
语料 100 万条
├── 分片 1 (10万) → FAISS IVF
├── 分片 2 (10万) → FAISS IVF
├── ...
└── 分片 10 (10万) → FAISS IVF

查询时 → LLM 判断相关分片 → 仅检索 1-2 个分片
```

适用于有明显领域划分的语料。

#### 方案 B：分层摘要（Map-Reduce RAG）

```
语料 100 万条
├── 叶节点片段 (chunks)
├── 中间层摘要 (每 10 个片段 → LLM 摘要)
└── 顶层摘要 (全局概览)

查询时 → ① 检索顶层索引 ② 定位到中间层 ③ 检索叶子层
```

#### 方案 C：替换向量库

| 方案 | 查询延迟 (1M) | 部署复杂度 |
|------|-------------|-----------|
| FAISS IVF + PQ | ~10ms | 低 |
| Chroma | ~20ms | 低 |
| Qdrant | ~5ms | 中 |
| Milvus | ~3ms | 高 |
| pgvector (PostgreSQL) | ~20ms | 中 |
| Elasticsearch | ~50ms | 中 |

#### 方案 D：缓存 + 异步

```
用户提问
  ├─ ① 查询缓存（LRU） → 命中直接返回
  └─ ② 未命中 → 异步检索 → LLM 流式输出 → 写入缓存
```

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def cached_retrieve(query: str) -> str:
    return retriever.invoke(query)
```

---

## 推荐演进路径

```
Phase 1（你现在）  ─→  Phase 2（本周）           ─→  Phase 3（下月）
                                                          
FAISS Flat         →  FAISS IVF + 持久化        →  Milvus / Qdrant
单机内存            →  保存到磁盘免重建           →  分布式向量库
纯向量检索           →  混合检索 + 重排序          →  多阶段检索
同步调用            →  Stream 流式输出           →  LRU 缓存 + 异步
单一分割            →  语义分割 + 自适应 chunk    →  分层摘要索引
```

## 最直接的三个建议（按投入产出比）

1. **立刻做**：打开 `Stream` 流式输出 — 修改 3 行代码，感知延迟从 3s → 0.3s
2. **本周做**：持久化 FAISS 索引 — 不再每启动重建一次，省掉索引时间
3. **语料破 10 万时做**：切换到 FAISS IVF + 混合检索（Ensemble Retriever）— 检索延迟降 10 倍，召回率提升

## 附录：各方案成本估算（100 万条语料）

| 方案 | 开发成本 | 硬件成本 | 延迟改善 | 召回率影响 |
|------|---------|---------|---------|-----------|
| 流式输出 | 0.5 人天 | 0 | 感知延迟 -90% | 不变 |
| FAISS IVF+SQ8 | 1 人天 | 0 | 查询 -90% | -2% |
| 持久化索引 | 0.5 人天 | 0 | 启动 -100% | 不变 |
| LRU 缓存 | 1 人天 | 内存 +0.5G | 热门 -100% | 不变 |
| 混合检索+重排序 | 3 人天 | 内存 +2G | 查询 -20% | +15-30% |
| Milvus 分布式 | 10 人天 | 3 台 16G 服务器 | 查询 -95% | 不变 |
| 分层摘要索引 | 15 人天 (含调优) | LLM API 费用 ×1.5 | 查询 -50% | +5% |

---

## License

MIT
