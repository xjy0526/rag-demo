"""
app.py — 基于 Streamlit 的 Multi-Demo。
"""

import html as html_lib
import os
import sys
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(Path(__file__).parent))

for d in ["./data/chroma_db", "./data/uploads", "./data/extracted/images", "./data/extracted/tables"]:
    os.makedirs(d, exist_ok=True)


def get_api_keys() -> dict:
    """Load API keys from page input, environment variables, .env, or Streamlit secrets."""
    keys = {
        "qwen": st.session_state.get("qwen_key_input", "")
        or os.getenv("DASHSCOPE_API_KEY", "")
        or os.getenv("QWEN_API_KEY", ""),
        "github": st.session_state.get("github_key_input", "") or os.getenv("GITHUB_TOKEN", ""),
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


def _has_streamlit_secret(*names: str) -> bool:
    """Safely check Streamlit secrets without requiring a secrets file."""
    try:
        return any(bool(st.secrets.get(name)) for name in names)
    except Exception:
        return False


def _reset_chat_if_source_changed(source_id: str) -> None:
    previous = st.session_state.get("current_source_id")
    if previous != source_id:
        st.session_state["rag_messages"] = []
    st.session_state["current_source_id"] = source_id


def render_workflow_grid(items: list[dict]) -> None:
    cards = []
    for idx, item in enumerate(items, start=1):
        kicker = html_lib.escape(item.get("kicker", f"{idx:02d}"))
        title = html_lib.escape(item["title"])
        body = html_lib.escape(item["body"])
        tone = html_lib.escape(item.get("tone", "blue"))
        cards.append(
            f"""
<article class="workflow-card tone-{tone}">
    <div class="workflow-step">{kicker}</div>
    <div class="workflow-title">{title}</div>
    <div class="workflow-text">{body}</div>
</article>
"""
        )
    st.markdown(f'<div class="workflow-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_note_card(kicker: str, title: str, body: str, tone: str = "blue") -> None:
    st.markdown(
        f"""
<div class="note-card tone-{html_lib.escape(tone)}">
    <div class="note-kicker">{html_lib.escape(kicker)}</div>
    <div class="note-title">{html_lib.escape(title)}</div>
    <div class="note-body">{html_lib.escape(body)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero_section():
    st.markdown(
        """
<section class="page-shell">
    <div class="hero-eyebrow">Multi-Modal AI Demo</div>
    <h1 class="page-title">Multi-Demo</h1>
    <p class="page-lead">
        一个面向展示与体验的 AI Demo 页面，
        支持 PDF 问答、哔哩哔哩视频解析，以及多来源 AI 研究检索。
    </p>
    <div class="hero-chip-row">
        <span class="hero-chip">Qwen Native</span>
        <span class="hero-chip">Bilibili Ready</span>
        <span class="hero-chip">Research Search</span>
        <span class="hero-chip">Streamlit + LangGraph</span>
    </div>
    <div class="hero-stat-row">
        <div class="hero-stat">
            <div class="hero-stat-value">PDF + 视频</div>
            <div class="hero-stat-label">文档与视频内容问答</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">多源检索</div>
            <div class="hero-stat-label">论文 图书 仓库 网站 视频</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">中文优先</div>
            <div class="hero-stat-label">研究摘要可切换英文</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">Qwen 驱动</div>
            <div class="hero-stat-label">多模态理解与生成</div>
        </div>
    </div>
    <div class="hero-glance-grid">
        <article class="glance-card">
            <div class="glance-kicker">Document QA</div>
            <strong>上传 PDF 后，直接围绕文档内容发起问答。</strong>
            <p>支持文本、图片与表格解析，适合展示多模态文档理解能力。</p>
        </article>
        <article class="glance-card">
            <div class="glance-kicker">Research Search</div>
            <strong>输入一个主题，快速查看多来源研究结果。</strong>
            <p>系统会并行检索论文、图书、GitHub 仓库、网站和视频，并生成摘要。</p>
        </article>
        <article class="glance-card">
            <div class="glance-kicker">Visual Demo</div>
            <strong>更清晰的展示布局，更适合直接作为演示页面使用。</strong>
            <p>界面围绕浏览、输入、检索和结果展示做了统一设计，更适合公开展示。</p>
        </article>
    </div>
</section>
""",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #eaf0f8;
        --surface: #ffffff;
        --surface-strong: #f7faff;
        --surface-soft: #eef4fb;
        --ink: #111827;
        --ink-soft: #334155;
        --muted: #5f6b7b;
        --line: rgba(148, 163, 184, 0.28);
        --line-strong: rgba(100, 116, 139, 0.34);
        --brand: #111111;
        --accent: #0a84ff;
        --accent-soft: #e7f1ff;
        --shadow: 0 24px 64px rgba(15, 23, 42, 0.12);
        --shadow-soft: 0 14px 36px rgba(15, 23, 42, 0.1);
    }

    @keyframes riseIn {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    html,
    body,
    [class*="css"] {
        font-family: "Instrument Sans", "Noto Sans SC", "PingFang SC", sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(10, 132, 255, 0.11), transparent 18%),
            radial-gradient(circle at 88% 12%, rgba(94, 92, 230, 0.09), transparent 18%),
            radial-gradient(circle at 50% 100%, rgba(255, 159, 10, 0.05), transparent 18%),
            linear-gradient(180deg, #edf3fb 0%, #e6edf7 100%);
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1320px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp div[data-testid="stMarkdownContainer"] p {
        font-size: 1.22rem;
        line-height: 1.8;
        color: #334155;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        letter-spacing: -0.04em;
        color: var(--ink);
    }

    .page-shell {
        margin-bottom: 2.1rem;
        padding: 4.6rem 4rem 3.4rem;
        border-radius: 34px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background:
            linear-gradient(135deg, #ffffff, #eef5ff),
            radial-gradient(circle at 85% 20%, rgba(10, 132, 255, 0.08), transparent 20%);
        box-shadow: var(--shadow);
        animation: riseIn 0.45s ease-out both;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.95rem;
        border-radius: 999px;
        border: 1px solid rgba(10, 132, 255, 0.16);
        background: #eef5ff;
        color: #2563eb;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        box-shadow: 0 8px 20px rgba(10, 132, 255, 0.08);
    }

    .page-title {
        max-width: 10ch;
        margin: 1rem 0 0.85rem;
        font-size: clamp(4.2rem, 8vw, 6.6rem);
        line-height: 0.92;
        color: #0f172a;
        font-weight: 800;
    }

    .page-lead {
        max-width: 760px;
        margin: 0;
        color: #5f6672;
        font-size: 1.56rem;
        line-height: 1.8;
    }

    .hero-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.9rem;
        margin-top: 2rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.86rem 1.15rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
        color: #111827;
        font-size: 1.08rem;
        font-weight: 700;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }

    .hero-chip:nth-child(1) {
        background: #eef6ff;
    }

    .hero-chip:nth-child(2) {
        background: #f3eeff;
    }

    .hero-chip:nth-child(3) {
        background: #edf9f1;
    }

    .hero-chip:nth-child(4) {
        background: #fff6ea;
    }

    .hero-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.11);
    }

    .hero-stat-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.35rem;
    }

    .hero-stat {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .hero-stat-value {
        margin-bottom: 0.22rem;
        color: #0f172a;
        font-size: 1rem;
        font-weight: 800;
    }

    .hero-stat-label {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .hero-glance-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.35rem;
        margin-top: 2.35rem;
    }

    .glance-card {
        min-height: 100%;
        padding: 1.8rem;
        border-radius: 30px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: #ffffff;
        box-shadow: var(--shadow-soft);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        animation: riseIn 0.45s ease-out both;
        position: relative;
    }

    .hero-glance-grid .glance-card:nth-child(1) {
        background: linear-gradient(180deg, #eef6ff, #ffffff);
    }

    .hero-glance-grid .glance-card:nth-child(2) {
        background: linear-gradient(180deg, #f4efff, #ffffff);
    }

    .hero-glance-grid .glance-card:nth-child(3) {
        background: linear-gradient(180deg, #fff4e8, #ffffff);
    }

    .glance-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 24px 54px rgba(15, 23, 42, 0.12);
    }

    .glance-kicker {
        margin-bottom: 0.9rem;
        color: #6b7280;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .glance-card strong {
        display: block;
        margin-bottom: 0.7rem;
        color: #0f172a;
        font-size: 1.62rem;
        line-height: 1.4;
        font-weight: 800;
    }

    .glance-card p {
        margin: 0;
        color: #5f6672 !important;
        font-size: 1.12rem !important;
        line-height: 1.82 !important;
    }

    .top-card {
        min-height: 100%;
        margin: 0 0 1.35rem;
        padding: 1.95rem 1.95rem 2.1rem;
        border-radius: 32px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: #ffffff;
        box-shadow: var(--shadow-soft);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        animation: riseIn 0.45s ease-out both;
        position: relative;
    }

    .top-card:nth-of-type(1) {
        background: linear-gradient(180deg, #eef6ff, #ffffff);
    }

    .top-card:nth-of-type(2) {
        background: linear-gradient(180deg, #f4efff, #ffffff);
    }

    .top-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 24px 54px rgba(15, 23, 42, 0.12);
    }

    .card-kicker {
        display: inline-flex;
        align-items: center;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: #f5f8fc;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .top-card h3 {
        margin: 1.15rem 0 1rem;
        font-size: 2.28rem;
        line-height: 1.22;
        color: #0f172a;
    }

    .top-card p {
        margin: 0 0 1rem;
        color: #5f6672 !important;
        font-size: 1.18rem !important;
        line-height: 1.82 !important;
    }

    .apple-list {
        margin: 1.15rem 0 0;
        padding: 0;
        list-style: none;
    }

    .apple-list li {
        position: relative;
        margin: 0.8rem 0;
        padding-left: 1.15rem;
        color: #374151;
        font-size: 1.14rem;
        line-height: 1.72;
    }

    .apple-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.68rem;
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #0a84ff, #5e5ce6);
    }

    .section-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0 0 1.25rem;
        padding: 0.64rem 1.02rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: #f1f6fc;
        color: #4d5b6d;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .section-intro {
        margin: 0 0 1.8rem;
        padding: 1.6rem 1.7rem;
        border-radius: 32px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: linear-gradient(180deg, #f8fbff, #eff5fc);
        box-shadow: var(--shadow-soft);
        color: #415062 !important;
        font-size: 1.2rem !important;
        line-height: 1.9 !important;
        animation: riseIn 0.45s ease-out both;
        position: relative;
    }

    .section-intro p {
        margin: 0;
        color: #415062 !important;
        font-size: 1.2rem !important;
        line-height: 1.9 !important;
    }

    .section-intro strong {
        color: #0f172a;
    }

    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 0 0 1.8rem;
    }

    .workflow-card {
        padding: 1.25rem 1.25rem 1.3rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: #ffffff;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .workflow-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
    }

    .workflow-card.tone-blue {
        background: linear-gradient(180deg, #f1f7ff, #ffffff);
    }

    .workflow-card.tone-purple {
        background: linear-gradient(180deg, #f5f1ff, #ffffff);
    }

    .workflow-card.tone-green {
        background: linear-gradient(180deg, #f1fbf5, #ffffff);
    }

    .workflow-card.tone-amber {
        background: linear-gradient(180deg, #fff8ef, #ffffff);
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2.25rem;
        height: 2.25rem;
        padding: 0 0.68rem;
        margin-bottom: 0.8rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.07);
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .workflow-title {
        margin-bottom: 0.48rem;
        color: #0f172a;
        font-size: 1.18rem;
        line-height: 1.42;
        font-weight: 800;
    }

    .workflow-text {
        color: #526072;
        font-size: 1rem;
        line-height: 1.72;
    }

    .note-card {
        margin: 0 0 1rem;
        padding: 1.15rem 1.2rem 1.2rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: #ffffff;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .note-card.tone-blue {
        background: linear-gradient(180deg, #f2f8ff, #ffffff);
    }

    .note-card.tone-purple {
        background: linear-gradient(180deg, #f5f0ff, #ffffff);
    }

    .note-card.tone-green {
        background: linear-gradient(180deg, #f1fbf5, #ffffff);
    }

    .note-card.tone-amber {
        background: linear-gradient(180deg, #fff8ef, #ffffff);
    }

    .note-kicker {
        margin-bottom: 0.48rem;
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .note-title {
        margin-bottom: 0.45rem;
        color: #0f172a;
        font-size: 1.16rem;
        line-height: 1.42;
        font-weight: 800;
    }

    .note-body {
        color: #526072;
        font-size: 1rem;
        line-height: 1.72;
    }

    .option-help {
        margin-top: -0.1rem;
        padding-left: 0.1rem;
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.52;
    }

    .chat-header-card {
        margin: 0 0 1rem;
        padding: 1.15rem 1.2rem 1.2rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, #f4f8ff, #ffffff);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .chat-header-title {
        margin-bottom: 0.4rem;
        color: #0f172a;
        font-size: 1.22rem;
        font-weight: 800;
    }

    .chat-header-meta {
        color: #526072;
        font-size: 1rem;
        line-height: 1.68;
    }

    div[data-testid="stTabs"] {
        margin-top: 1.25rem;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.72rem;
        padding: 0.56rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: #edf3fa;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stTabs"] button {
        font-size: 1.12rem;
        font-weight: 700;
    }

    div[data-testid="stTabs"] [role="tab"] {
        min-height: 3.95rem;
        padding: 0 1.45rem;
        border-radius: 999px;
        color: #4f5f72;
        transition: all 0.18s ease;
    }

    div[data-testid="stTabs"] [role="tab"]:hover {
        color: #1f2937;
        background: rgba(255, 255, 255, 0.66);
    }

    div[data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, #e9f3ff, #f2edff);
        color: #111827;
        box-shadow: 0 12px 26px rgba(10, 132, 255, 0.18);
    }

    div[data-testid="stForm"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"],
    div[data-testid="stMetric"],
    [data-testid="stChatMessage"],
    .stAlert {
        border-radius: 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: #ffffff;
        box-shadow: var(--shadow-soft);
    }

    div[data-testid="stForm"],
    div[data-testid="stExpander"] {
        padding: 0.65rem 0.8rem;
    }

    div[data-testid="stExpander"] {
        overflow: hidden;
        margin-bottom: 1.25rem;
    }

    details {
        border-radius: 26px;
    }

    div[data-testid="stMetric"] {
        padding: 1.2rem 1.2rem 1.28rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 42px rgba(15, 23, 42, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 2.8rem;
        font-weight: 800;
    }

    .stTextInput label,
    .stSlider label,
    .stFileUploader label,
    .stSelectbox label,
    .stRadio label {
        color: #111827;
        font-size: 1.16rem !important;
        font-weight: 700 !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        min-height: 4.15rem;
        padding: 0.95rem 1.15rem !important;
        border-radius: 22px !important;
        border: 1px solid var(--line-strong) !important;
        background: #ffffff !important;
        color: #111827 !important;
        font-size: 1.24rem !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
        transition: box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(10, 132, 255, 0.45) !important;
        box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.12) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    .stChatInput input::placeholder {
        color: #9aa1ad !important;
        font-size: 1.12rem !important;
        opacity: 1 !important;
    }

    .stTextInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"],
    .stChatInput div[data-baseweb="base-input"] {
        border-radius: 22px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        background: transparent !important;
    }

    .stCaption,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp .stMarkdown small,
    .stFileUploader small,
    .stTextInput div[data-baseweb="input"] + div,
    .stSlider p {
        color: #7b8190 !important;
        font-size: 1.02rem !important;
        line-height: 1.72 !important;
    }

    .stCheckbox {
        padding: 0.38rem 0;
    }

    .stCheckbox label,
    .stCheckbox p {
        color: #1f2937 !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        line-height: 1.58 !important;
    }

    .stCheckbox [data-baseweb="checkbox"] {
        transform: scale(1.18);
        transform-origin: left center;
        margin-right: 0.7rem !important;
    }

    .stCheckbox [data-baseweb="checkbox"] > div {
        border-radius: 0.4rem !important;
    }

    .stRadio [role="radiogroup"] label {
        gap: 0.9rem !important;
    }

    .stRadio input[type="radio"] {
        transform: scale(1.22);
        accent-color: #111827;
    }

    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {
        min-height: 4.1rem;
        padding: 0.96rem 1.8rem !important;
        border-radius: 999px !important;
        border: 1px solid rgba(17, 24, 39, 0.06) !important;
        font-size: 1.14rem !important;
        font-weight: 700 !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f172a 0%, #0f3d91 58%, #5e5ce6 100%);
        color: white;
        box-shadow: 0 14px 28px rgba(58, 76, 196, 0.22);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(58, 76, 196, 0.26);
    }

    .stChatInput input {
        font-size: 1.2rem !important;
    }

    .stChatInput div[data-baseweb="base-input"] {
        min-height: 4.8rem !important;
        padding: 0.72rem 1rem !important;
        border-radius: 24px !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: #ffffff !important;
        box-shadow: var(--shadow-soft);
    }

    .stFileUploader section {
        padding: 1.25rem !important;
        border-radius: 26px !important;
    }

    button[title],
    [data-testid="stTooltipIcon"] button,
    [data-testid="stBaseButton-help"],
    .stTextInput button,
    .stChatInput button,
    .stNumberInput button,
    .stSelectbox button {
        min-width: 2.8rem !important;
        min-height: 2.8rem !important;
    }

    button[title] svg,
    [data-testid="stTooltipIcon"] svg,
    [data-testid="stBaseButton-help"] svg,
    .stTextInput button svg,
    .stChatInput button svg,
    .stNumberInput button svg,
    .stSelectbox button svg {
        width: 1.1rem !important;
        height: 1.1rem !important;
    }

    .stAlert {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    .stAlert p,
    .stAlert div {
        color: #334155 !important;
        font-size: 1.08rem !important;
        line-height: 1.76 !important;
    }

    [data-testid="stChatMessage"] {
        padding: 1.2rem 1.3rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.12);
    }

    .page-shell::after,
    .glance-card::after,
    .top-card::after,
    .section-intro::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {
        font-size: 1.12rem !important;
        line-height: 1.84 !important;
    }

    hr {
        margin: 2.6rem 0 !important;
        border-top: 1px solid rgba(15, 23, 42, 0.08) !important;
    }

    @media (max-width: 1100px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.2rem;
        }

        .page-shell {
            padding: 3.4rem 1.5rem 2.6rem;
            border-radius: 32px;
        }

        .page-title {
            font-size: clamp(3.4rem, 12vw, 4.8rem);
        }

        .page-lead {
            font-size: 1.24rem;
        }

        .hero-glance-grid {
            grid-template-columns: 1fr;
        }

        .hero-stat-row {
            grid-template-columns: 1fr 1fr;
        }

        .workflow-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

def render_top_panels(keys: dict):
    col_info, col_config = st.columns([1.35, 1], gap="large")

    with col_info:
        st.markdown(
            """
<div class="top-card">
    <div class="card-kicker">Demo Overview</div>
    <h3>一个集中展示文档问答、视频解析与研究检索能力的 AI Demo 页面。</h3>
    <p>
        页面分为两个主要能力区域：文档与视频问答，以及 AI 研究探索。
        你可以直接上传内容、输入链接或发起主题检索，快速体验完整流程。
    </p>
    <ul class="apple-list">
        <li>支持 PDF 文本、图片与表格的多模态解析。</li>
        <li>支持哔哩哔哩视频链接的字幕与元信息分析。</li>
        <li>支持 AI / ML 主题的多来源研究检索与摘要生成。</li>
    </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_config:
        st.markdown(
            """
<div class="top-card">
    <div class="card-kicker">Run Setup</div>
    <h3>填写运行所需的 API 配置后，即可开始体验主要功能。</h3>
    <p>
        支持从 Streamlit Secrets 自动加载，也支持在当前页面手动输入。
        配置完成后，可以直接体验索引、问答和研究检索流程。
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        qwen_from_secrets = _has_streamlit_secret("DASHSCOPE_API_KEY", "QWEN_API_KEY")
        qwen_from_env = bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"))
        github_from_secrets = _has_streamlit_secret("GITHUB_TOKEN")
        github_from_env = bool(os.getenv("GITHUB_TOKEN"))

        if qwen_from_secrets or qwen_from_env:
            source = "Streamlit Secrets" if qwen_from_secrets else ".env / 环境变量"
            st.success(f"已从 {source} 加载千问密钥")
        else:
            st.session_state["qwen_key_input"] = st.text_input(
                "千问 API Key *",
                value=keys["qwen"],
                type="password",
                help="填写 DASHSCOPE_API_KEY 或兼容的 Qwen API Key。",
            )

        if github_from_secrets or github_from_env:
            source = "Streamlit Secrets" if github_from_secrets else ".env / 环境变量"
            st.info(f"已从 {source} 加载 GitHub Token")
        else:
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

    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.markdown(
            f"""
<div class="chat-header-card">
    <div class="note-kicker">Chat Session</div>
    <div class="chat-header-title">基于当前内容继续提问</div>
    <div class="chat-header-meta">当前内容：{html_lib.escape(source_name)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with action_col:
        if st.session_state.get("current_source_url"):
            st.link_button("查看源链接", st.session_state["current_source_url"], use_container_width=True)

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
            with st.spinner("正在检索相关内容……"):
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
                        with st.expander("查看检索内容"):
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
    render_workflow_grid(
        [
            {"kicker": "01", "title": "导入内容", "body": "上传 PDF 或粘贴哔哩哔哩链接，开始本轮内容分析。", "tone": "blue"},
            {"kicker": "02", "title": "生成索引", "body": "系统会整理文本、图片、表格或字幕，并建立检索数据。", "tone": "purple"},
            {"kicker": "03", "title": "发起问答", "body": "索引完成后可以直接提问，查看基于当前内容的回答结果。", "tone": "amber"},
        ]
    )

    valid, err = validate_keys(keys)

    tab_pdf, tab_bilibili = st.tabs(["📄 PDF 文档", "📺 哔哩哔哩视频"])

    with tab_pdf:
        col_upload, col_options = st.columns([2.8, 2.2])
        with col_upload:
            render_note_card("PDF Demo", "适合论文、报告、教材与手册。", "上传后会自动抽取文本、图片与表格，用于后续展示问答能力。", tone="blue")
            uploaded_file = st.file_uploader(
                "上传 PDF 文档",
                type=["pdf"],
                help="支持论文、报告、手册、书籍等 PDF 文件。",
            )

        with col_options:
            render_note_card("Index Options", "按需关闭部分处理步骤。", "如果只需要演示正文问答，可以跳过图片或表格处理来加快索引。", tone="purple")
            st.markdown("**索引选项**")
            option_col1, option_col2 = st.columns(2, gap="medium")
            with option_col1:
                skip_images = st.checkbox("跳过图片理解", value=False, key="skip_images")
                st.markdown('<div class="option-help">更快，但不会理解图表和插图。</div>', unsafe_allow_html=True)
            with option_col2:
                skip_tables = st.checkbox("跳过表格处理", value=False, key="skip_tables")
                st.markdown('<div class="option-help">更快，但不会理解结构化表格信息。</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            doc_name = Path(uploaded_file.name).stem
            upload_path = f"./data/uploads/{uploaded_file.name}"
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            do_index = st.button("索引 PDF", type="primary", use_container_width=True, key="index_pdf")
            if do_index:
                if not valid:
                    st.warning(f"{err}。将先建立可检索索引；问答生成前仍需要配置千问 API Key。")
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

                        has_indexed_content = any(
                            stats.get(key, 0) > 0 for key in ("text_count", "image_count", "table_count")
                        )
                        if has_indexed_content:
                            st.success("PDF 索引完成")
                        else:
                            st.warning("PDF 已处理，但没有索引到可用内容。请检查文件是否为扫描件、加密文件或空白文档。")
                        render_index_stats(
                            stats.get("text_count", 0),
                            stats.get("image_count", 0),
                            stats.get("table_count", 0),
                        )
                    except Exception as e:
                        st.error(f"索引失败：{e}")
                        st.code(traceback.format_exc())

    with tab_bilibili:
        render_note_card("Bilibili Demo", "优先使用字幕与元信息建立内容分析。", "适合课程视频、讲解视频与资讯内容。字幕越完整，展示效果通常越好。", tone="amber")
        st.markdown("输入一个哔哩哔哩视频链接，系统会提取字幕与视频信息，并接入后续问答。")
        bilibili_url = st.text_input(
            "哔哩哔哩视频链接",
            placeholder="例如：https://www.bilibili.com/video/BV1xx411c7mD",
        )
        do_parse = st.button("解析并索引视频", type="primary", use_container_width=True, key="index_bilibili")

        if do_parse:
            if not bilibili_url.strip():
                st.warning("请先输入哔哩哔哩视频链接。")
            else:
                if not valid:
                    st.warning(f"{err}。视频解析和索引会继续执行；后续问答生成前仍需要配置千问 API Key。")
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
    render_workflow_grid(
        [
            {"kicker": "Refine", "title": "输入主题", "body": "主题越具体，检索出的论文、仓库、网站和视频通常越聚焦。", "tone": "blue"},
            {"kicker": "Search", "title": "执行检索", "body": "系统会同时查看论文、图书、GitHub、网站和哔哩哔哩内容。", "tone": "green"},
            {"kicker": "Summary", "title": "查看摘要", "body": "千问会基于多来源结果生成一份更适合浏览的主题概览。", "tone": "purple"},
        ]
    )

    output_language = "en" if output_language_label == "English" else "zh"

    hint_col1, hint_col2 = st.columns(2)
    with hint_col1:
        render_note_card("Query Design", "主题越具体，结果通常越好。", "可以直接写问题、任务目标，或者一个明确的模型 / 方法方向。", tone="blue")
    with hint_col2:
        render_note_card("Priority Hints", "你可以给系统一些偏好条件。", "例如指定优先 UP 主、仓库 URL 或论文标题，让结果更贴近展示目标。", tone="green")

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

        progress_bar = st.progress(0, text="开始检索……")
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
                render_note_card(
                    "Summary",
                    "下面是由千问整理的结果摘要。",
                    "可以先浏览整体概览，再查看论文、仓库、网站和视频等分类结果。",
                    tone="purple" if output_language == "zh" else "blue",
                )
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
    render_hero_section()
    keys = get_api_keys()
    render_top_panels(keys)

    tab_rag, tab_research = st.tabs(["📄 文档与视频问答", "🔬 研究探索"])
    with tab_rag:
        render_rag_tab(keys)
    with tab_research:
        render_research_tab(keys)


if __name__ == "__main__":
    main()
