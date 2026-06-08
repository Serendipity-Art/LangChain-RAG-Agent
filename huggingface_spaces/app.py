"""
LangChain RAG Agent — HuggingFace Spaces Gradio UI
===================================================
基于 @LangChain_RAG_Agent.ipynb 构建的可视化问答界面
支持：Excel/CSV 上传 → 向量库构建 → RAG 问答 + Agent 模式
"""

import os
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def build_llm(api_key: str, base_url: str, model_name: str, temperature: float):
    """创建 LLM 实例"""
    if not api_key:
        return None
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
    )


def build_embeddings(model_name: str):
    """创建 Embedding 模型"""
    return HuggingFaceEmbeddings(model_name=model_name)


def load_dataframe(file_obj) -> pd.DataFrame:
    """
    从上传文件加载 DataFrame。
    支持 .xlsx / .xls / .csv，自动识别编码。
    Gradio 6 兼容：file_obj 可能是字符串路径或 FileData 对象。
    """
    if file_obj is None:
        return None

    # Gradio 6 type="filepath" 返回字符串，Gradio 5 返回对象含 .name
    if isinstance(file_obj, str):
        filepath = file_obj
    elif hasattr(file_obj, "name"):
        filepath = file_obj.name
    else:
        raise ValueError("无法识别的文件输入格式")

    suffix = Path(filepath).suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    elif suffix == ".csv":
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="gbk")
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，请上传 Excel (.xlsx) 或 CSV (.csv)")

    return df


def build_vector_store(
    df: pd.DataFrame,
    embedding_model: HuggingFaceEmbeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    """
    将 DataFrame → Document 列表 → 文本分割 → FAISS 向量库。
    """
    # Step 1: 每行转为一个 Document
    docs = []
    for _, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=content, metadata={"source": "upload"}))

    # Step 2: 文本分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("分割后没有产生任何文档片段，请检查数据内容")

    # Step 3: 构建 FAISS 索引
    vs = FAISS.from_documents(chunks, embedding_model)
    return vs


# ──────────────────────────────────────────────
# RAG 问答核心逻辑
# ──────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """你是一个回答机器人。你的任务是根据下述给定的已知信息回答用户问题。

严格要求：
1. 确保你的回复完全基于下面的已知信息，不编造答案。
2. 如果已知信息不足以回答用户的问题，请直接回复"我无法回答你的问题"。
3. 不要使用任何 Markdown 格式，请用纯文本回复。

已知信息：
{info}

用户问：{question}

请用中文回答用户问题："""


def answer_rag(query: str, vector_store: FAISS, llm: ChatOpenAI, k: int = 3):
    """
    纯 RAG 模式：检索 → 拼接 → LLM 回答
    返回 (retrieval_time, llm_time, total_time, answer, retrieved_docs)
    """
    t0 = time.time()

    # 1. 检索
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    t1 = time.time()
    retrieval_time = round(t1 - t0, 3)

    # 2. 构建 prompt
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(
        info=retrieved_text, question=query
    )

    # 3. LLM 回答
    response = llm.invoke(prompt)
    answer = response.content

    t2 = time.time()
    llm_time = round(t2 - t1, 3)
    total_time = round(t2 - t0, 3)

    return retrieval_time, llm_time, total_time, answer, retrieved_docs


def answer_agent(query: str, vector_store: FAISS, llm: ChatOpenAI, k: int = 3):
    """
    Agent 模式：用 LangChain create_agent 包裹检索工具。
    传入预构建的 llm 实例解决 API Key 动态配置的问题。
    """
    from langchain.agents import create_agent
    from langchain.tools import tool

    @tool
    def search_corpus(q: str) -> str:
        """搜索本地语料库中与问题相关的信息"""
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(q)
        return "\n\n".join([doc.page_content for doc in docs])

    # 传入 llm 实例而非字符串，避免重新走 init_chat_model
    agent = create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=(
            "你是一个有帮助的助手，可以根据检索到的内容回答用户问题。"
            "不要使用任何 markdown 格式，用纯文本回复。"
            "如果检索到的信息不足以回答，请说'我无法回答你的问题'。"
        ),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # agent 返回的 messages 最后一条是 assistant 消息
    last_msg = result["messages"][-1]
    # 兼容不同版本的 content 格式
    if hasattr(last_msg, "content"):
        return last_msg.content
    elif isinstance(last_msg, dict) and "content" in last_msg:
        return last_msg["content"]
    elif "content_blocks" in last_msg:
        return last_msg["content_blocks"][0]["text"]
    return str(last_msg)


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

def process_upload(file_obj, chunk_size, chunk_overlap, embedding_model_name,
                   progress=gr.Progress()):
    """上传文件 → 构建向量库"""
    if file_obj is None:
        return None, "⚠️ 请先选择一个文件"

    progress(0, desc="读取文件...")
    try:
        df = load_dataframe(file_obj)
    except Exception as e:
        return None, f"❌ 文件读取失败：{e}"
    if df is None or df.empty:
        return None, "⚠️ 文件为空"

    progress(0.2, desc="构建 Embedding 模型...")
    emb = build_embeddings(embedding_model_name)

    progress(0.4, desc="分割文档({}条)...".format(len(df)))
    try:
        vs = build_vector_store(df, emb, chunk_size, chunk_overlap)
    except Exception as e:
        return None, f"❌ 向量库构建失败：{e}"

    progress(0.9, desc="向量库就绪 ✓")
    n_chunks = vs.index.ntotal if hasattr(vs, "index") else "?"

    return vs, f"✅ 成功！{len(df)} 条记录 → {n_chunks} 个片段 → 向量库已就绪"


def answer_question(query, vector_store_state, api_key, base_url, model_name,
                    temperature, k, mode):
    """回答问题（RAG 或 Agent 模式）"""
    if vector_store_state is None:
        return "⚠️ 请先上传数据构建向量库", "", "", ""

    if not api_key:
        return "⚠️ 请在左侧设置 API Key", "", "", ""

    llm = build_llm(api_key, base_url, model_name, temperature)
    if llm is None:
        return "⚠️ LLM 初始化失败，请检查 API Key", "", "", ""

    try:
        if mode == "Agent 模式":
            t0 = time.time()
            answer = answer_agent(query, vector_store_state, llm, k)
            elapsed = round(time.time() - t0, 3)
            stats = f"⏱️ 总耗时: {elapsed} 秒 (Agent 模式)"
            docs_info = "（Agent 模式下检索过程由 LLM 自行调度，暂不显示详细文档片段）"
            return answer, stats, docs_info, ""
        else:
            ret_t, llm_t, total_t, answer, retrieved = answer_rag(
                query, vector_store_state, llm, k
            )
            stats = (
                f"🔍 检索耗时: {ret_t} 秒\n"
                f"🧠 LLM 推理耗时: {llm_t} 秒\n"
                f"⏱️ 总耗时: {total_t} 秒\n"
                f"📄 检索片段数: {len(retrieved)}"
            )
            docs_info = "---\n\n".join(
                [f"**片段 {i+1}** (前 200 字):\n{doc.page_content[:200]}..."
                 for i, doc in enumerate(retrieved)]
            )
            return answer, stats, docs_info, ""
    except Exception as e:
        return f"❌ 回答生成失败：{e}", "", "", ""


def build_ui():
    """构建 Gradio 界面"""
    with gr.Blocks(title="LangChain RAG Agent") as demo:
        gr.Markdown(
            """
            # 🦜 LangChain RAG Agent — 可视化问答系统

            基于 `LangChain + FAISS + DeepSeek` 的检索增强生成系统。
            上传你的 Excel/CSV 语料 → 自动构建向量库 → 提问即可获得基于语料的回答。
            """
        )

        # ── 状态 ──
        vector_store_state = gr.State(None)

        # ── 推荐：Tab 布局，桌面/手机都友好 ──
        with gr.Tabs():
            with gr.Tab("⚙️ 配置"):
                with gr.Accordion("🔑 API 设置", open=False):
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="sk-... / HF Spaces secrets 中设置",
                        value=os.getenv("OPENAI_API_KEY1", ""),
                    )
                    base_url = gr.Textbox(
                        label="Base URL",
                        placeholder="https://api.deepseek.com/v1",
                        value=os.getenv("OPENAI_BASE_URL1",
                                         "https://api.deepseek.com/v1"),
                    )
                    model_name = gr.Dropdown(
                        label="模型",
                        choices=[
                            "deepseek-v4-flash", "deepseek-v4-pro",
                            "gpt-4o-mini", "gpt-4o",
                        ],
                        value="deepseek-v4-flash",
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0, maximum=2, value=0.7, step=0.1,
                    )

                with gr.Accordion("📂 数据与向量库", open=True):
                    file_input = gr.File(
                        label="上传语料文件 (.xlsx / .csv)",
                        file_types=[".xlsx", ".xls", ".csv"],
                    )
                    with gr.Row():
                        chunk_size = gr.Slider(
                            label="Chunk Size",
                            minimum=200, maximum=2000, value=500, step=50,
                        )
                        chunk_overlap = gr.Slider(
                            label="Overlap",
                            minimum=0, maximum=400, value=50, step=10,
                        )
                    embedding_model = gr.Dropdown(
                        label="Embedding 模型",
                        choices=[
                            "all-MiniLM-L6-v2",
                            "all-mpnet-base-v2",
                            "BAAI/bge-small-zh-v1.5",
                            "BAAI/bge-base-zh-v1.5",
                        ],
                        value="all-MiniLM-L6-v2",
                    )
                    build_btn = gr.Button(
                        "🚀 构建向量库", variant="primary"
                    )

                with gr.Accordion("🔍 检索设置", open=True):
                    k_slider = gr.Slider(
                        label="检索 Top-K", minimum=1, maximum=10,
                        value=3, step=1,
                    )
                    mode_radio = gr.Radio(
                        label="回答模式",
                        choices=["标准 RAG", "Agent 模式"],
                        value="标准 RAG",
                    )

                build_status = gr.Markdown("💡 请上传语料文件构建向量库")

            # ── Tab 2：问答 ──
            with gr.Tab("💬 问答"):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=400,
                    avatar_images=[None, None],
                )
                query_input = gr.Textbox(
                    label="输入你的问题",
                    placeholder="例如：苏州园林有什么特点？",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button(
                        "🎯 发送", variant="primary", scale=2
                    )
                    clear_btn = gr.Button("🔄 清空对话", scale=1)

                gr.Markdown("### 📊 性能统计")
                stats_output = gr.Markdown("", elem_classes="stats-box")

                with gr.Accordion("📄 检索到的文档片段", open=False):
                    docs_display = gr.Markdown("", elem_classes="doc-box")

        # ── 事件 ──

        # 构建向量库
        build_btn.click(
            fn=process_upload,
            inputs=[file_input, chunk_size, chunk_overlap, embedding_model],
            outputs=[vector_store_state, build_status],
        ).then(
            fn=lambda: [],
            outputs=[chatbot],
        )

        # 问答
        def respond(query, history, vs_state, api_key_val, base_url_val,
                    model_val, temp_val, k_val, mode_val):
            if not query or not query.strip():
                return history, "", ""

            answer, stats, docs, err = answer_question(
                query, vs_state, api_key_val, base_url_val,
                model_val, temp_val, k_val, mode_val,
            )

            if err and err.startswith("⚠️"):
                history = history or []
                history.append({"role": "user", "content": [{"type": "text", "text": query}]})
                history.append({"role": "assistant", "content": [{"type": "text", "text": f"❌ {err}"}]})
                return history, "", ""

            history = history or []
            history.append({"role": "user", "content": [{"type": "text", "text": query}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
            return history, stats, docs

        submit_btn.click(
            fn=respond,
            inputs=[
                query_input, chatbot, vector_store_state,
                api_key, base_url, model_name, temperature,
                k_slider, mode_radio,
            ],
            outputs=[chatbot, stats_output, docs_display],
        ).then(
            fn=lambda: "",
            outputs=[query_input],
        )

        query_input.submit(
            fn=respond,
            inputs=[
                query_input, chatbot, vector_store_state,
                api_key, base_url, model_name, temperature,
                k_slider, mode_radio,
            ],
            outputs=[chatbot, stats_output, docs_display],
        ).then(
            fn=lambda: "",
            outputs=[query_input],
        )

        clear_btn.click(
            fn=lambda: ([], "", ""),
            outputs=[chatbot, stats_output, docs_display],
        )

        # 示例问题
        gr.Markdown("### 💡 快速开始")
        gr.Examples(
            examples=[
                ["苏州园林有什么特点？"],
                ["虎丘塔的历史背景是什么？"],
                ["苏州园林和北方园林有什么区别？"],
                ["拙政园是谁建造的？"],
            ],
            inputs=query_input,
        )

    return demo


# ──────────────────────────────────────────────
# 启动
# ──────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_port=port,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
        css="""
        .stats-box { font-size: 14px; }
        .doc-box {
            max-height: 300px;
            overflow-y: auto;
            background: #f5f5f5;
            padding: 10px;
            border-radius: 8px;
            font-size: 13px;
        }
        """
    )
