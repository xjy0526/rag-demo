"""
config.py — Central configuration for the Multimodal RAG Research Assistant.
"""

import os
import streamlit as st


def get_config() -> dict:
    """Load configuration from env vars or Streamlit secrets."""
    config = {}

    try:
        config["qwen_api_key"] = (
            st.secrets.get("DASHSCOPE_API_KEY")
            or st.secrets.get("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY", "")
            or os.getenv("QWEN_API_KEY", "")
        )
        config["github_token"] = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
    except Exception:
        config["qwen_api_key"] = os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")
        config["github_token"] = os.getenv("GITHUB_TOKEN", "")

    config["qwen_base_url"] = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return config


# Qwen settings
QWEN_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
QWEN_REASONING_MODEL = os.getenv("QWEN_REASONING_MODEL", QWEN_TEXT_MODEL)
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-flash-2026-01-22")

# ChromaDB settings
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_TEXT = "text_chunks"
COLLECTION_IMAGES = "image_captions"
COLLECTION_TABLES = "table_descriptions"

# Extraction dirs
IMAGES_DIR = "./data/extracted/images"
TABLES_DIR = "./data/extracted/tables"
UPLOADS_DIR = "./data/uploads"

# Search result limits
MAX_PAPERS = 20
MAX_BOOKS = 10
MAX_REPOS = 20
TOP_DISPLAY = 5

EMBED_MODEL = "all-MiniLM-L6-v2"
