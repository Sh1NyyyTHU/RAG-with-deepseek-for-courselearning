"""
Streamlit UI — 课程课件智能问答与做题助手

Run: streamlit run app.py
"""
import streamlit as st
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, normalize_latex
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.deepseek_client import DeepSeekClient
from src.knowledge_base import KnowledgeBase
from src.history_manager import HistoryManager
from src.ocr_adapter import OCRAdapter
from src.prompt_templates import MODE_CONFIG
import config

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="课程课件智能问答与做题助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ───────────────────────────────
def init_session_state():
    defaults = {
        "embedding_service": None,
        "vector_store": None,
        "retriever": None,
        "deepseek_client": None,
        "knowledge_base": None,
        "history_manager": None,
        "ocr_adapter": None,
        "initialized": False,
        "api_connected": False,
        "embedding_ready": False,
        "qa_history": [],
        "last_answer": None,
        "last_sources": None,
        "last_chunks": None,
        "last_mode": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ── Initialize services ─────────────────────────────────────────
@st.cache_resource
def get_embedding_service():
    svc = EmbeddingService()
    svc.initialize()
    return svc

@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_ocr_adapter():
    return OCRAdapter()


def get_or_create_services():
    if st.session_state.embedding_service is None:
        st.session_state.embedding_service = get_embedding_service()
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_vector_store()
    if st.session_state.ocr_adapter is None:
        st.session_state.ocr_adapter = get_ocr_adapter()

    if st.session_state.knowledge_base is None:
        st.session_state.knowledge_base = KnowledgeBase(
            st.session_state.embedding_service,
            st.session_state.vector_store,
        )

    if st.session_state.retriever is None:
        st.session_state.retriever = Retriever(
            st.session_state.embedding_service,
            st.session_state.vector_store,
            top_k=st.session_state.get("top_k", config.DEFAULT_TOP_K),
        )

    if st.session_state.deepseek_client is None:
        st.session_state.deepseek_client = DeepSeekClient()

    if st.session_state.history_manager is None:
        st.session_state.history_manager = HistoryManager()

    st.session_state.initialized = True


get_or_create_services()

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 系统设置")

    # Model settings
    with st.expander("🤖 DeepSeek 模型设置", expanded=False):
        st.text(f"模型: {config.DEEPSEEK_MODEL}")
        st.text(f"API 地址: {config.DEEPSEEK_BASE_URL}")

        if st.button("🔌 测试 API 连通性", use_container_width=True):
            with st.spinner("测试中..."):
                result = st.session_state.deepseek_client.check_connectivity()
                if result["status"] == "ok":
                    st.success("✅ API 连接正常")
                    st.session_state.api_connected = True
                else:
                    st.error(f"❌ 连接失败: {result.get('error', '未知错误')}")
                    st.session_state.api_connected = False

        if st.session_state.api_connected:
            st.success("API 已连接")
        else:
            st.warning("API 未验证")

    # Embedding status
    with st.expander("🧮 Embedding 状态", expanded=False):
        status = st.session_state.embedding_service.get_status()
        st.text(f"模型: {status['model_name']}")
        st.text(f"设备: {status['device']}")
        st.text(f"FP16: {status['use_fp16']}")
        st.text(f"CUDA 可用: {status['cuda_available']}")
        if status["initialized"]:
            st.success("✅ Embedding 已就绪")
            st.session_state.embedding_ready = True
        else:
            st.error("❌ Embedding 未就绪")

    # Retrieval settings
    with st.expander("🔍 检索设置", expanded=False):
        top_k = st.slider("top_k", 1, 30, config.DEFAULT_TOP_K, key="top_k")
        chunk_size = st.number_input(
            "chunk_size", 200, 2000, config.CHUNK_SIZE, 50, key="chunk_size"
        )
        chunk_overlap = st.number_input(
            "chunk_overlap", 0, 500, config.CHUNK_OVERLAP, 10, key="chunk_overlap"
        )
        show_retrieved = st.checkbox("显示检索片段", value=False, key="show_retrieved")

        # Update retriever top_k
        if st.session_state.retriever:
            st.session_state.retriever.top_k = top_k

    # Answer mode
    st.divider()
    st.subheader("📝 回答模式")
    mode = st.radio(
        "选择模式",
        options=["qa", "teaching", "solving"],
        format_func=lambda x: MODE_CONFIG[x]["label"],
        key="answer_mode",
    )

    st.divider()

    # PDF filter
    st.subheader("📂 检索范围")
    if st.session_state.vector_store:
        files = st.session_state.vector_store.get_indexed_files()
        file_names = [f["file_name"] for f in files]
        if file_names:
            filter_all = st.checkbox("检索所有文件", value=True, key="filter_all")
            if not filter_all:
                selected_files = st.multiselect(
                    "选择要检索的文件",
                    options=file_names,
                    default=file_names[:5] if len(file_names) > 5 else file_names,
                    key="selected_files",
                )
            else:
                selected_files = None
        else:
            selected_files = None
            st.info("暂无已索引文件")
    else:
        selected_files = None

# ── Title ───────────────────────────────────────────────────────
st.title("📚 课程课件智能问答与做题助手")
st.caption(f"高电压工程课程学习助手 | 当前模式: {MODE_CONFIG[mode]['label']}")

# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 知识库管理", "💬 问答与做题", "📋 历史记录与系统状态"])

# ── TAB 1: Knowledge Base Management ────────────────────────────
with tab1:
    st.header("知识库管理")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("上传 PDF 课件")
        uploaded_files = st.file_uploader(
            "选择一个或多个 PDF 文件",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            if st.button("📥 建立索引", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="准备中...")
                status_container = st.container()

                total = len(uploaded_files)
                results = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save to data/pdfs
                    dest_path = config.PDF_DIR / uploaded_file.name
                    if dest_path.exists():
                        stem = dest_path.stem
                        suffix = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = config.PDF_DIR / f"{stem}_{counter}{suffix}"
                            counter += 1

                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    progress_bar.progress(
                        (idx + 0.5) / total,
                        text=f"正在索引: {uploaded_file.name} ({idx+1}/{total})",
                    )

                    def progress_cb(step, total_steps):
                        progress_bar.progress(
                            (idx + step / total_steps) / total,
                            text=f"索引中: {uploaded_file.name} (步骤 {step}/{total_steps})",
                        )

                    result = st.session_state.knowledge_base.index_pdf(
                        dest_path,
                        chunk_size=st.session_state.get("chunk_size", config.CHUNK_SIZE),
                        chunk_overlap=st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP),
                        progress_callback=progress_cb,
                    )
                    results.append(result)

                progress_bar.progress(1.0, text="完成！")

                # Show results
                for r in results:
                    if r["status"] == "ok":
                        status_container.success(
                            f"✅ {r['file_name']}: {r['pages']} 页, {r['chunks']} 个片段"
                        )
                        if r.get("warnings"):
                            for w in r["warnings"]:
                                status_container.warning(w)
                    elif r["status"] == "skipped":
                        status_container.info(f"⏭️ {r['file_name']}: 已索引，跳过")
                    else:
                        status_container.error(
                            f"❌ {r['file_name']}: {r.get('error', '未知错误')}"
                        )

                # Clear cache to reflect new files
                st.cache_resource.clear()
                st.rerun()

    with col2:
        st.subheader("批量操作")
        if st.button("🗑️ 清空知识库", use_container_width=True, type="secondary"):
            st.session_state.knowledge_base.clear()
            st.success("知识库已清空")
            st.cache_resource.clear()
            st.rerun()

    # File list
    st.divider()
    st.subheader("📋 当前已索引文件")

    files = st.session_state.vector_store.get_indexed_files()
    if files:
        for f in files:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.text(f["file_name"])
                with col2:
                    st.text(f"{f.get('pages', '?')} 页")
                with col3:
                    st.text(f"{f.get('chunks', '?')} chunks")
                with col4:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔄", key=f"reindex_{f['file_hash'][:8]}", help="重新索引"):
                            # Find the original PDF
                            pdf_path = config.PDF_DIR / f["file_name"]
                            if pdf_path.exists():
                                with st.spinner(f"重新索引 {f['file_name']}..."):
                                    result = st.session_state.knowledge_base.reindex_pdf(pdf_path)
                                    if result["status"] == "ok":
                                        st.success(f"已重新索引: {result['chunks']} 个片段")
                                    else:
                                        st.error(result.get("error", "失败"))
                                st.rerun()
                            else:
                                st.error("源文件不存在")
                    with c2:
                        if st.button("🗑️", key=f"del_{f['file_hash'][:8]}", help="删除"):
                            st.session_state.knowledge_base.delete_pdf(f["file_hash"])
                            st.success(f"已删除: {f['file_name']}")
                            st.rerun()
                st.caption(f"Hash: {f['file_hash'][:16]}... | 索引时间: {f.get('indexed_at', 'unknown')}")
                st.divider()
    else:
        st.info("暂无已索引文件。请上传 PDF 并建立索引。")

# ── TAB 2: Q&A ──────────────────────────────────────────────────
with tab2:
    st.header("问答与做题")

    # Image upload for OCR
    with st.expander("📷 题目图片上传（可选）", expanded=False):
        uploaded_image = st.file_uploader(
            "上传题目截图 (PNG/JPG/JPEG)",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
        )
        if uploaded_image:
            ocr = st.session_state.ocr_adapter
            if ocr.available:
                with st.spinner("正在识别图片文字..."):
                    text = ocr.extract_image_text(uploaded_image.getvalue())
                    if text:
                        st.success("文字识别完成，已填入输入框")
                        st.session_state["ocr_text"] = text
                    else:
                        st.warning("未能识别到文字")
            else:
                st.warning("PaddleOCR 未安装。请手动将题目文字复制到下方输入框。")
                st.info("安装 PaddleOCR: pip install paddlepaddle paddleocr")

    # Question input
    default_text = st.session_state.get("ocr_text", "")
    question = st.text_area(
        "输入你的问题或题目：",
        value=default_text,
        height=150,
        placeholder="例如：\n- 什么是汤逊放电理论？\n- 请讲解巴申定律\n- 已知某变压器额定电压为220kV/110kV...",
        key="question_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🚀 提交", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ 清空输入", use_container_width=False):
            st.session_state.question_input = ""
            st.session_state.ocr_text = ""
            st.rerun()

    if submit and question.strip():
        if not st.session_state.embedding_ready:
            st.error("Embedding 模型未就绪，请检查日志。")
        else:
            with st.spinner("🔍 正在检索课件..."):
                # Get file filter
                file_filter = None
                if not st.session_state.get("filter_all", True):
                    file_filter = st.session_state.get("selected_files", None)

                chunks = st.session_state.retriever.retrieve(
                    query=question,
                    file_filter=file_filter,
                )

                if not chunks:
                    st.warning("⚠️ 未检索到相关课件片段，将仅使用模型知识回答。")

            # Build prompt
            mode_cfg = MODE_CONFIG[mode]
            user_prompt = mode_cfg["build_user_prompt"](question, chunks)
            system_prompt = mode_cfg["system"]

            # Display response
            st.divider()
            st.subheader(f"📝 回答 ({mode_cfg['label']})")

            try:
                full_response = [""]  # use list for mutable closure
                stream = st.session_state.deepseek_client.ask(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mode=mode,
                    stream=True,
                )

                # Use st.write_stream for proper markdown streaming
                def response_generator():
                    for chunk in stream:
                        chunk = normalize_latex(chunk)
                        full_response[0] += chunk
                        yield chunk

                st.write_stream(response_generator)

                # Save to history
                st.session_state.history_manager.add(
                    question=question,
                    answer=full_response[0],
                    mode=mode,
                    sources=chunks,
                    retrieved_chunks=chunks,
                )

                st.session_state.last_answer = full_response[0]
                st.session_state.last_sources = chunks
                st.session_state.last_chunks = chunks
                st.session_state.last_mode = mode

            except ConnectionError as e:
                st.error(f"🔌 连接错误: {e}")
            except TimeoutError as e:
                st.error(f"⏱️ 超时: {e}")
            except RuntimeError as e:
                st.error(f"❌ API 错误: {e}")
            except Exception as e:
                st.error(f"❌ 未知错误: {e}")

            # Show sources
            if chunks:
                with st.expander("📖 引用来源", expanded=True):
                    seen = set()
                    for c in chunks:
                        key = (c["file_name"], c["page_number"])
                        if key not in seen:
                            seen.add(key)
                            st.markdown(
                                f"- **{c['file_name']}**，第 **{c['page_number']}** 页 "
                                f"(相似度: {c['similarity']:.3f})"
                            )

            # Show retrieved chunks (if enabled)
            if st.session_state.get("show_retrieved") and chunks:
                with st.expander("🔎 检索到的课件片段", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**片段 {i}** — {c['file_name']}，第 {c['page_number']} 页 "
                                    f"(相似度: {c['similarity']:.3f})")
                        st.text_area(
                            f"chunk_{i}",
                            value=c["text"],
                            height=100,
                            key=f"chunk_display_{i}",
                            label_visibility="collapsed",
                        )

    # Show last answer if exists (for copy)
    if st.session_state.last_answer and not submit:
        st.divider()
        st.subheader(f"📝 上一次回答 ({MODE_CONFIG[st.session_state.last_mode]['label']})")
        st.markdown(st.session_state.last_answer)

# ── TAB 3: History & System Status ──────────────────────────────
with tab3:
    st.header("历史记录与系统状态")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📜 问答历史")

        records = st.session_state.history_manager.get_all()
        if not records:
            st.info("暂无问答记录")

        for i, record in enumerate(records):
            with st.expander(
                f"Q{i+1}: {record['question'][:60]}... ({record['mode']}) — {record['timestamp'][:19]}",
                expanded=False,
            ):
                st.markdown(f"**问题**：{record['question']}")
                st.markdown(f"**模式**：{record['mode']}")
                st.markdown(f"**回答**：\n{record['answer']}")
                if record.get("sources"):
                    st.markdown("**引用来源**：")
                    for src in record["sources"]:
                        st.markdown(f"- {src['file_name']}，第 {src['page_number']} 页")

    with col2:
        st.subheader("📊 系统状态")

        # KB status
        kb_status = st.session_state.knowledge_base.get_status()
        st.metric("已索引文件数", kb_status["file_count"])
        st.metric("Chunk 总数", kb_status["total_chunks"])

        # Embedding
        emb_status = st.session_state.embedding_service.get_status()
        st.metric("Embedding 设备", emb_status["device"])
        st.metric("CUDA 可用", "✅" if emb_status["cuda_available"] else "❌")

        # API
        st.metric("DeepSeek 模型", config.DEEPSEEK_MODEL)

        if st.button("📤 导出问答记录为 Markdown", use_container_width=True):
            filepath = st.session_state.history_manager.save_export()
            st.success(f"已导出到: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇️ 下载导出文件",
                    data=f.read(),
                    file_name=filepath.name,
                    mime="text/markdown",
                )

        if st.button("🗑️ 清空历史记录", use_container_width=True):
            st.session_state.history_manager.clear()
            st.rerun()

# ── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption(f"Courseware QA System | Embedding: {config.EMBEDDING_MODEL_NAME} | "
           f"LLM: {config.DEEPSEEK_MODEL} | Vector DB: ChromaDB")
