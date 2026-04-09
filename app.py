"""
app.py — 基于 Streamlit 的多模态 RAG 研究助手。
"""

import os
import sys
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="多模态 RAG 研究助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(Path(__file__).parent))

for d in ["./data/chroma_db", "./data/uploads", "./data/extracted/images", "./data/extracted/tables"]:
    os.makedirs(d, exist_ok=True)


def get_api_keys() -> dict:
    """Load API keys from Streamlit secrets or session state inputs."""
    keys = {
        "qwen": st.session_state.get("qwen_key_input", ""),
        "github": st.session_state.get("github_key_input", ""),
    }
    try:
        if st.secrets.get("DASHSCOPE_API_KEY"):
            keys["qwen"] = st.secrets["DASHSCOPE_API_KEY"]
        elif st.secrets.get("QWEN_API_KEY"):
            keys["qwen"] = st.secrets["QWEN_API_KEY"]
        if st.secrets.get("GITHUB_TOKEN"):
            keys["github"] = st.secrets["GITHUB_TOKEN"]
    except Exception:
        pass
    return keys


def validate_keys(keys: dict) -> tuple[bool, str]:
    """Check that required API keys are present."""
    if not keys["qwen"]:
        return False, "缺少千问 API Key（DASHSCOPE_API_KEY）"
    return True, ""


def _reset_chat_if_source_changed(source_id: str) -> None:
    previous = st.session_state.get("current_source_id")
    if previous != source_id:
        st.session_state["rag_messages"] = []
    st.session_state["current_source_id"] = source_id


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    :root {
        --bg: #f6f1e8;
        --surface: rgba(255, 252, 247, 0.88);
        --surface-strong: #fffdf8;
        --ink: #1f2937;
        --muted: #5b6475;
        --line: rgba(123, 97, 63, 0.16);
        --brand: #c8553d;
        --brand-deep: #8f2d23;
        --accent: #e5b94b;
        --shadow: 0 18px 55px rgba(58, 36, 18, 0.10);
    }

    html, body, [class*="css"]  {
        font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(229, 185, 75, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(200, 85, 61, 0.14), transparent 26%),
            linear-gradient(180deg, #fbf7f0 0%, #f3ede3 100%);
        color: var(--ink);
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp div[data-testid="stMarkdownContainer"] p {
        font-size: 3rem;
        line-height: 1.45;
        color: var(--ink);
    }

    .stApp h1, .stApp h2, .stApp h3 {
        letter-spacing: -0.02em;
    }

    .page-shell {
        margin-bottom: 1.5rem;
    }

    .page-title {
        font-size: clamp(3.8rem, 5vw, 5.8rem);
        line-height: 1.05;
        margin: 0 0 0.8rem;
        color: #1d2430;
        font-weight: 800;
    }

    .page-lead {
        margin: 0;
        max-width: 1000px;
        color: var(--muted);
        font-size: 2rem;
        line-height: 1.55;
    }

    .top-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.4rem 1.45rem;
        margin: 0 0 1rem;
        box-shadow: 0 10px 30px rgba(60, 45, 30, 0.05);
    }

    .top-card h3 {
        margin: 0 0 0.75rem;
        font-size: 2rem;
        color: #1b2430;
    }

    .top-card p {
        font-size: 3.6rem;
        line-height: 1.42;
    }

    .section-intro {
        background: rgba(255, 252, 247, 0.76);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin: 0 0 1.25rem;
        font-size: 3rem !important;
        line-height: 1.45 !important;
    }

    .section-intro p {
        font-size: 3rem !important;
        line-height: 1.45 !important;
        margin: 0;
    }

    .top-card strong,
    .section-intro strong {
        color: var(--brand-deep);
    }

    .section-header {
        display: inline-block;
        border-bottom: 3px solid rgba(200, 85, 61, 0.26);
        padding-bottom: 0.32rem;
        margin-bottom: 0.8rem;
        color: #1b2430;
        font-size: 3.4rem;
        font-weight: 800;
    }

    div[data-testid="stTabs"] {
        margin-top: 1rem;
    }

    div[data-testid="stTabs"] button {
        font-size: 2rem;
        font-weight: 700;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid var(--line);
        padding: 0.4rem;
        border-radius: 16px;
    }

    div[data-testid="stTabs"] [role="tab"] {
        border-radius: 12px;
        min-height: 86px;
        padding: 0 1.4rem;
    }

    div[data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(200, 85, 61, 0.12), rgba(229, 185, 75, 0.18));
        color: var(--brand-deep);
    }

    div[data-testid="stForm"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"],
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 8px 28px rgba(58, 36, 18, 0.05);
    }

    div[data-testid="stExpander"] {
        overflow: hidden;
        margin-bottom: 0.9rem;
    }

    details {
        border-radius: 18px;
    }

    div[data-testid="stMetric"] {
        padding: 0.4rem 0.25rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1.8rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 3rem;
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        font-size: 4rem !important;
        min-height: 7.2rem;
        border-radius: 20px !important;
        border: 1px solid rgba(123, 97, 63, 0.18) !important;
        background: rgba(255, 253, 248, 0.95) !important;
        padding: 1rem 1.2rem !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        font-size: 3.2rem !important;
        color: #7a7f8c !important;
        opacity: 1 !important;
    }

    .stTextInput label,
    .stSlider label,
    .stFileUploader label {
        font-size: 4rem !important;
        font-weight: 700 !important;
    }

    .stCaption,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp .stMarkdown small {
        font-size: 3rem !important;
        line-height: 1.4 !important;
        color: var(--muted) !important;
    }

    .stFileUploader small,
    .stTextInput div[data-baseweb="input"] + div,
    .stSlider p {
        font-size: 3rem !important;
    }

    .stCheckbox {
        padding: 0.8rem 0;
    }

    .stCheckbox [data-testid="stCheckbox"] > label,
    .stCheckbox label {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        gap: 1rem !important;
        font-size: 1.45rem !important;
        font-weight: 600 !important;
    }

    .stCheckbox p {
        font-size: 1.45rem !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }

    .stCheckbox [data-testid="stTooltipIcon"],
    .stCheckbox [data-testid="stBaseButton-help"] {
        display: inline-flex !important;
        align-items: center !important;
        align-self: center !important;
        margin-left: 0.3rem !important;
        flex: 0 0 auto !important;
    }

    .stCheckbox [data-testid="stTooltipIcon"] button,
    .stCheckbox [data-testid="stBaseButton-help"] {
        min-width: 2.4rem !important;
        min-height: 2.4rem !important;
    }

    .stCheckbox [data-testid="stTooltipIcon"] svg,
    .stCheckbox [data-testid="stBaseButton-help"] svg {
        width: 1.5rem !important;
        height: 1.5rem !important;
    }

    .stCheckbox [data-baseweb="checkbox"] {
        transform: scale(2.4);
        transform-origin: left center;
        margin-right: 1.6rem !important;
    }

    .stCheckbox [data-baseweb="checkbox"] > div {
        width: 2.2rem !important;
        height: 2.2rem !important;
        border-radius: 0.45rem !important;
    }

    .stCheckbox [data-baseweb="checkbox"] svg {
        width: 1.6rem !important;
        height: 1.6rem !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {
        min-height: 7rem;
        border-radius: 20px !important;
        font-size: 3.1rem !important;
        font-weight: 700 !important;
        padding: 1.2rem 1.8rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #c8553d 0%, #ae3e30 100%);
        color: white;
        border: none;
        box-shadow: 0 12px 22px rgba(200, 85, 61, 0.22);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(200, 85, 61, 0.22);
    }

    .stAlert {
        border-radius: 18px;
        font-size: 3.2rem;
    }

    .stAlert p,
    .stAlert div {
        font-size: 3.2rem !important;
        line-height: 1.4 !important;
    }

    .stSelectbox label,
    .stRadio label {
        font-size: 3.8rem !important;
    }

    .stRadio [role="radiogroup"] label {
        gap: 1rem !important;
    }

    .stRadio input[type="radio"] {
        transform: scale(2.5);
        accent-color: var(--brand);
    }

    .stChatInput input {
        font-size: 3.8rem !important;
        min-height: 9rem !important;
    }

    .stTextInput div[data-baseweb="input"],
    .stChatInput div[data-baseweb="base-input"],
    .stTextArea textarea {
        min-height: 7rem !important;
    }

    .stChatInput div[data-baseweb="base-input"] {
        min-height: 9rem !important;
        border-radius: 24px !important;
        padding: 0.8rem 1rem !important;
    }

    .stChatInput input::placeholder {
        font-size: 3.2rem !important;
    }

    .stFileUploader section {
        padding: 1.2rem !important;
        border-radius: 20px !important;
    }

    button[title],
    [data-testid="stTooltipIcon"] button,
    [data-testid="stBaseButton-help"] {
        min-width: 4.2rem !important;
        min-height: 4.2rem !important;
    }

    button[title] svg,
    [data-testid="stTooltipIcon"] svg,
    [data-testid="stBaseButton-help"] svg,
    .stTextInput button svg,
    .stChatInput button svg,
    .stNumberInput button svg,
    .stSelectbox button svg {
        width: 2.8rem !important;
        height: 2.8rem !important;
    }

    .stTextInput button,
    .stChatInput button,
    .stNumberInput button,
    .stSelectbox button {
        min-width: 4.4rem !important;
        min-height: 4.4rem !important;
    }

    .stTextInput [role="button"],
    .stChatInput [role="button"] {
        transform: scale(1.8);
        transform-origin: center;
    }

    [data-testid="stMetricLabel"] {
        font-size: 2.8rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 4rem;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 252, 247, 0.82);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 0.8rem 1rem;
    }

    @media (max-width: 1100px) {
        .page-lead {
            font-size: 2rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

def render_top_panels(keys: dict):
    col_info, col_config = st.columns([1.6, 1], gap="large")

    with col_info:
        st.markdown(
            """
<div class="top-card">
    <h3>项目说明</h3>
    <p><strong>文档与视频问答</strong>：上传 PDF 或粘贴哔哩哔哩视频链接，建立索引后直接进行问答。</p>
    <p><strong>研究探索</strong>：围绕一个 AI / ML 主题，同时检索论文、图书、GitHub 仓库、网站和视频。</p>
    <p><strong>底层能力</strong>：千问负责推理与图像理解，ChromaDB 负责检索，LangGraph 负责流程编排。</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_config:
        st.markdown(
            """
<div class="top-card">
    <h3>运行配置</h3>
</div>
""",
            unsafe_allow_html=True,
        )

        keys_from_secrets = False
        try:
            keys_from_secrets = bool(st.secrets.get("DASHSCOPE_API_KEY") or st.secrets.get("QWEN_API_KEY"))
        except Exception:
            keys_from_secrets = False

        if keys_from_secrets:
            st.success("已从 Streamlit Secrets 加载千问密钥")
            if keys.get("github"):
                st.info("GitHub Token 已从 Secrets 加载")
        else:
            st.session_state["qwen_key_input"] = st.text_input(
                "千问 API Key *",
                value=keys["qwen"],
                type="password",
                help="填写 DASHSCOPE_API_KEY 或兼容的 Qwen API Key。",
            )
            st.session_state["github_key_input"] = st.text_input(
                "GitHub Token（可选）",
                value=keys["github"],
                type="password",
                help="可提升 GitHub API 速率限制。",
            )


def render_chat_section(keys: dict):
    source_name = st.session_state.get("current_source_name")
    if not source_name:
        st.info("先上传 PDF，或先解析一个哔哩哔哩视频链接，然后再开始提问。")
        return

    valid, err = validate_keys(keys)

    st.markdown("### 💬 开始提问")
    st.caption(f"当前知识源：**{source_name}**")
    if st.session_state.get("current_source_url"):
        st.link_button("查看源链接", st.session_state["current_source_url"], use_container_width=False)

    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("请输入你想了解的问题……")
    if query:
        if not valid:
            st.error(err)
            return

        st.session_state["rag_messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("正在检索文本、图片和表格上下文……"):
                try:
                    from src.rag_pipeline import answer_query
                    from src.ui_components import render_retrieved_context

                    result = answer_query(query=query, qwen_api_key=keys["qwen"])
                    answer = result["answer"]
                    query_types = result.get("query_types", [])
                    retrieved = result.get("retrieved_results", [])

                    if query_types:
                        badges = " ".join(f"`{item}`" for item in query_types)
                        st.caption(f"检索模态：{badges}")

                    st.markdown(answer)

                    if retrieved:
                        with st.expander("查看召回上下文"):
                            render_retrieved_context(retrieved)

                    st.session_state["rag_messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"❌ 回答失败：{e}"
                    st.error(err_msg)
                    st.code(traceback.format_exc())
                    st.session_state["rag_messages"].append({"role": "assistant", "content": err_msg})

    if st.session_state.get("rag_messages"):
        if st.button("清空对话历史"):
            st.session_state["rag_messages"] = []
            st.rerun()


def render_rag_tab(keys: dict):
    st.markdown('<h2 class="section-header">📄 文档与视频问答</h2>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-intro">
    上传 <strong>PDF</strong>，或者粘贴一个 <strong>哔哩哔哩视频链接</strong>。
    系统会先建立索引，再把文本、图片、表格或字幕内容接入问答。
</div>
""",
        unsafe_allow_html=True,
    )

    valid, err = validate_keys(keys)

    tab_pdf, tab_bilibili = st.tabs(["📄 PDF 文档", "📺 哔哩哔哩视频"])

    with tab_pdf:
        col_upload, col_options = st.columns([3, 2])
        with col_upload:
            uploaded_file = st.file_uploader(
                "上传 PDF 文档",
                type=["pdf"],
                help="支持论文、报告、手册、书籍等 PDF 文件。",
            )

        with col_options:
            st.markdown("**索引选项**")
            skip_images = st.checkbox("跳过图片理解", value=False, key="skip_images")
            st.caption("更快，但不会理解图表和插图。")

            skip_tables = st.checkbox("跳过表格处理", value=False, key="skip_tables")
            st.caption("更快，但不会理解结构化表格信息。")

        if uploaded_file is not None:
            doc_name = Path(uploaded_file.name).stem
            upload_path = f"./data/uploads/{uploaded_file.name}"
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            do_index = st.button("索引 PDF", type="primary", use_container_width=True, key="index_pdf")
            if do_index:
                if not valid:
                    st.error(f"无法索引：{err}")
                else:
                    with st.spinner("正在解析并索引 PDF……"):
                        try:
                            from src.rag_pipeline import index_document
                            from src.ui_components import render_index_stats

                            stats = index_document(
                                file_path=upload_path,
                                doc_name=doc_name,
                                qwen_api_key=keys["qwen"],
                                skip_images=skip_images,
                                skip_tables=skip_tables,
                            )
                            if stats.get("error"):
                                st.warning(f"部分步骤执行异常：{stats['error']}")

                            _reset_chat_if_source_changed(f"pdf::{doc_name}")
                            st.session_state["current_source_name"] = doc_name
                            st.session_state["current_source_url"] = ""

                            st.success("PDF 索引完成")
                            render_index_stats(
                                stats.get("text_count", 0),
                                stats.get("image_count", 0),
                                stats.get("table_count", 0),
                            )
                        except Exception as e:
                            st.error(f"索引失败：{e}")
                            st.code(traceback.format_exc())

    with tab_bilibili:
        st.markdown("输入一个哔哩哔哩视频链接，系统会优先提取字幕与元信息，再接入当前 RAG 问答流程。")
        bilibili_url = st.text_input(
            "哔哩哔哩视频链接",
            placeholder="例如：https://www.bilibili.com/video/BV1xx411c7mD",
        )
        do_parse = st.button("解析并索引视频", type="primary", use_container_width=True, key="index_bilibili")

        if do_parse:
            if not bilibili_url.strip():
                st.warning("请先输入哔哩哔哩视频链接。")
            elif not valid:
                st.error(f"无法解析：{err}")
            else:
                with st.spinner("正在抓取视频元信息与字幕……"):
                    try:
                        from src.tools.bilibili_tool import normalize_bilibili_url
                        from src.rag_pipeline import index_bilibili_video

                        clean_url = normalize_bilibili_url(bilibili_url.strip())
                        stats = index_bilibili_video(clean_url, qwen_api_key=keys["qwen"])
                        if stats.get("error"):
                            st.error(f"视频索引失败：{stats['error']}")
                        else:
                            _reset_chat_if_source_changed(f"bilibili::{stats['source_url']}")
                            st.session_state["current_source_name"] = stats["title"]
                            st.session_state["current_source_url"] = stats["source_url"]

                            c1, c2, c3 = st.columns(3)
                            c1.metric("文本块", stats.get("text_count", 0))
                            c2.metric("字幕条数", stats.get("subtitle_count", 0))
                            c3.metric("播放量", stats.get("view_count", "N/A"))
                            st.success("哔哩哔哩视频已接入问答")
                            st.caption(
                                f"标题：{stats.get('title', '')} · UP主：{stats.get('owner', '')} · "
                                f"发布时间：{stats.get('published', '')} · 时长：{stats.get('duration', '')}"
                            )
                    except Exception as e:
                        st.error(f"解析失败：{e}")
                        st.code(traceback.format_exc())

    st.divider()
    render_chat_section(keys)


def render_research_tab(keys: dict):
    title_col, lang_col = st.columns([3.2, 1.1])
    with title_col:
        st.markdown('<h2 class="section-header">🔬 AI 研究探索</h2>', unsafe_allow_html=True)
    with lang_col:
        output_language_label = st.radio(
            "摘要语言",
            options=["中文", "English"],
            horizontal=True,
            index=0,
            key="research_output_language",
            label_visibility="visible",
        )

    st.markdown(
        """
<div class="section-intro">
    输入一个 AI / ML 主题后，系统会并行检索 <strong>论文、图书、GitHub 仓库、网站和哔哩哔哩视频</strong>，
    然后生成一份更适合阅读的研究摘要。
</div>
""",
        unsafe_allow_html=True,
    )

    output_language = "en" if output_language_label == "English" else "zh"

    with st.form("research_form"):
        st.markdown("### 🔍 检索主题")
        query = st.text_input(
            "主题或问题",
            placeholder="例如：Transformer 注意力机制、图神经网络药物发现、多模态 RAG",
            help="主题越具体，结果通常越好。",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**优先线索（可选）**")
            priority_channel = st.text_input(
                "📺 优先 UP 主",
                placeholder="例如：李沐、3Blue1Brown、Yannic Kilcher",
                help="若指定，会优先展示该 UP 主相关视频。",
            )
            priority_repo = st.text_input(
                "💻 GitHub 仓库 URL",
                placeholder="例如：https://github.com/huggingface/transformers",
                help="若指定，会在仓库结果中优先考虑它。",
            )
            priority_paper = st.text_input(
                "📄 论文标题 / 链接 / 关键词",
                placeholder="例如：Attention is All You Need",
                help="若指定，会在论文结果中优先考虑它。",
            )

        with col2:
            st.markdown("**结果数量**")
            max_papers = st.slider("论文数量", 5, 20, 10)
            max_repos = st.slider("仓库数量", 5, 20, 10)
            max_books = st.slider("图书数量", 3, 10, 5)
            top_display = st.slider("默认展示条数", 3, 10, 5)

        submitted = st.form_submit_button("开始全量检索", type="primary", use_container_width=True)

    if submitted and query.strip():
        valid, err = validate_keys(keys)
        if not valid:
            st.warning(f"⚠️ {err}，仍会执行检索，但 AI 摘要能力会受限。")

        progress_bar = st.progress(0, text="开始研究流程……")
        status_text = st.empty()

        try:
            from src.research_agent import run_research
            from src.ui_components import (
                render_books_section,
                render_papers_section,
                render_repos_section,
                render_videos_section,
                render_websites_section,
            )

            status_text.text("正在用千问优化检索主题……")
            progress_bar.progress(10, text="优化主题")

            with st.spinner("正在执行研究管线……"):
                result = run_research(
                    query=query.strip(),
                    qwen_api_key=keys["qwen"],
                    github_token=keys["github"],
                    output_language=output_language,
                    priority_channel=priority_channel,
                    priority_repo_url=priority_repo,
                    priority_paper_url=priority_paper,
                    max_papers=max_papers,
                    max_books=max_books,
                    max_repos=max_repos,
                )

            progress_bar.progress(100, text="研究完成")
            status_text.empty()

            refined_topic = result.get("topic", query)
            st.markdown(f"## 📊 检索结果：*{refined_topic}*")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("论文", len(result.get("papers", [])))
            c2.metric("图书", len(result.get("books", [])))
            c3.metric("仓库", len(result.get("repos", [])))
            c4.metric("网站", len(result.get("websites", [])))
            c5.metric("视频", len(result.get("videos", [])))

            if result.get("summary"):
                st.markdown("### 🤖 AI Research Summary" if output_language == "en" else "### 🤖 AI 研究摘要")
                st.info(result["summary"])

            st.divider()

            tab_papers, tab_books, tab_repos, tab_websites, tab_videos = st.tabs(
                [
                    f"📄 论文 ({len(result.get('papers', []))})",
                    f"📚 图书 ({len(result.get('books', []))})",
                    f"💻 仓库 ({len(result.get('repos', []))})",
                    f"🌐 网站 ({len(result.get('websites', []))})",
                    f"📺 视频 ({len(result.get('videos', []))})",
                ]
            )

            with tab_papers:
                st.caption("来源：ArXiv + Semantic Scholar")
                render_papers_section(result.get("papers", []), top_n=top_display)

            with tab_books:
                st.caption("优先展示免费资源，也会补充 Open Library / Google Books")
                render_books_section(result.get("books", []), top_n=top_display)

            with tab_repos:
                st.caption("优先官方仓库，并按星标、相关性排序")
                render_repos_section(result.get("repos", []), top_n=top_display)

            with tab_websites:
                st.caption("文档、课程、博客与工具资源")
                render_websites_section(result.get("websites", []), top_n=8)

            with tab_videos:
                st.caption("来自哔哩哔哩的教学视频，指定 UP 主会优先展示")
                render_videos_section(result.get("videos", []), top_n=top_display)

            st.session_state["last_research"] = result
            st.session_state["last_query"] = query

        except Exception as e:
            progress_bar.progress(0)
            status_text.empty()
            st.error(f"研究流程执行失败：{e}")
            with st.expander("调试信息"):
                st.code(traceback.format_exc())

    elif submitted and not query.strip():
        st.warning("请输入检索主题。")
    elif not submitted and st.session_state.get("last_research"):
        st.info(f"正在显示上一次结果：**{st.session_state.get('last_query', '未知主题')}**")


def main():
    st.markdown(
        """
    <div class="page-shell">
        <h1 class="page-title">多模态 RAG 研究助手</h1>
        <p class="page-lead">
            支持 PDF 多模态问答、哔哩哔哩视频分析，以及论文、图书、代码仓库、网站和视频的联合检索。
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    keys = get_api_keys()
    render_top_panels(keys)

    tab_rag, tab_research = st.tabs(["📄 文档与视频问答", "🔬 研究探索"])
    with tab_rag:
        render_rag_tab(keys)
    with tab_research:
        render_research_tab(keys)


if __name__ == "__main__":
    main()
