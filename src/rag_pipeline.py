"""
rag_pipeline.py — Orchestrates the multimodal RAG pipeline using LangGraph.
"""

from __future__ import annotations

import traceback
from typing import TypedDict


class RAGState(TypedDict):
    file_path: str
    doc_name: str
    text_blocks: list
    image_paths: list
    tables: list
    image_data: list
    table_data: list
    text_count: int
    image_count: int
    table_count: int
    query: str
    query_types: list
    retrieved_results: list
    answer: str
    error: str
    skip_images: bool
    skip_tables: bool
    qwen_api_key: str


def _default_rag_state(
    file_path: str = "",
    doc_name: str = "",
    qwen_api_key: str = "",
    skip_images: bool = False,
    skip_tables: bool = False,
) -> RAGState:
    return RAGState(
        file_path=file_path,
        doc_name=doc_name,
        text_blocks=[],
        image_paths=[],
        tables=[],
        image_data=[],
        table_data=[],
        text_count=0,
        image_count=0,
        table_count=0,
        query="",
        query_types=[],
        retrieved_results=[],
        answer="",
        error="",
        skip_images=skip_images,
        skip_tables=skip_tables,
        qwen_api_key=qwen_api_key,
    )


def node_parse(state: RAGState) -> RAGState:
    """Parse PDF into text, images, and tables."""
    try:
        from src.multimodal_parser import parse_document

        doc = parse_document(state["file_path"])
        state["text_blocks"] = doc.text_blocks
        state["image_paths"] = doc.image_paths
        state["tables"] = doc.tables
    except Exception as e:
        state["error"] = f"Parse failed: {e}"
        print(f"[rag_pipeline] Parse failed: {e}\n{traceback.format_exc()}")
    return state


def node_index_text(state: RAGState) -> RAGState:
    """Index text chunks into ChromaDB."""
    try:
        from src.indexer import index_text

        state["text_count"] = index_text(state["text_blocks"], state["doc_name"], reset=False)
    except Exception as e:
        print(f"[rag_pipeline] Text indexing failed: {e}")
    return state


def node_caption_images(state: RAGState) -> RAGState:
    """Caption and index images using Qwen VL."""
    if state.get("skip_images") or not state["image_paths"]:
        return state
    try:
        from src.indexer import caption_images_with_qwen, index_images

        image_data = caption_images_with_qwen(state["image_paths"], state["qwen_api_key"])
        state["image_data"] = image_data
        state["image_count"] = index_images(image_data, state["doc_name"])
    except Exception as e:
        print(f"[rag_pipeline] Image captioning failed: {e}")
    return state


def node_process_tables(state: RAGState) -> RAGState:
    """Process and index tables using Qwen."""
    if state.get("skip_tables") or not state["tables"]:
        return state
    try:
        from src.indexer import index_tables, process_tables
        from src.llm_clients import get_qwen_llm

        llm = get_qwen_llm(state["qwen_api_key"], temperature=0.1, max_tokens=2048)
        table_data = process_tables(state["tables"], llm, state["doc_name"])
        state["table_data"] = table_data
        state["table_count"] = index_tables(table_data, state["doc_name"])
    except Exception as e:
        print(f"[rag_pipeline] Table processing failed: {e}")
    return state


def node_route_query(state: RAGState) -> RAGState:
    """Classify the query using Qwen."""
    try:
        from src.llm_clients import get_qwen_llm
        from src.query_router import QueryType, classify_query

        llm = get_qwen_llm(state["qwen_api_key"], temperature=0.1, max_tokens=2048)
        state["query_types"] = classify_query(state["query"], llm)
    except Exception as e:
        print(f"[rag_pipeline] Query routing failed: {e}")
        from src.query_router import QueryType

        state["query_types"] = [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
    return state


def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant context from ChromaDB."""
    try:
        from src.retriever import retrieve_all

        state["retrieved_results"] = retrieve_all(state["query"], state["query_types"], k=4)
    except Exception as e:
        print(f"[rag_pipeline] Retrieval failed: {e}")
        state["retrieved_results"] = []
    return state


def node_generate(state: RAGState) -> RAGState:
    """Generate the final answer using Qwen."""
    try:
        from src.generator import generate_answer
        from src.llm_clients import get_qwen_llm

        llm = get_qwen_llm(state["qwen_api_key"])
        state["answer"] = generate_answer(
            state["query"],
            state["retrieved_results"],
            llm,
        )
    except Exception as e:
        state["answer"] = f"Answer generation failed: {e}"
        print(f"[rag_pipeline] Generation failed: {e}")
    return state


def build_indexing_graph():
    """Build LangGraph for document indexing."""
    try:
        from langgraph.graph import END, StateGraph

        builder = StateGraph(RAGState)
        builder.add_node("parse", node_parse)
        builder.add_node("index_text", node_index_text)
        builder.add_node("caption_images", node_caption_images)
        builder.add_node("process_tables", node_process_tables)
        builder.set_entry_point("parse")
        builder.add_edge("parse", "index_text")
        builder.add_edge("index_text", "caption_images")
        builder.add_edge("caption_images", "process_tables")
        builder.add_edge("process_tables", END)
        return builder.compile()
    except Exception as e:
        print(f"[rag_pipeline] Indexing graph build failed: {e}")
        return None


def build_query_graph():
    """Build LangGraph for query answering."""
    try:
        from langgraph.graph import END, StateGraph

        builder = StateGraph(RAGState)
        builder.add_node("route", node_route_query)
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("generate", node_generate)
        builder.set_entry_point("route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()
    except Exception as e:
        print(f"[rag_pipeline] Query graph build failed: {e}")
        return None


def index_document(
    file_path: str,
    doc_name: str,
    qwen_api_key: str,
    skip_images: bool = False,
    skip_tables: bool = False,
) -> dict:
    """
    Parse and index a PDF document. Returns stats dict.
    Falls back to sequential execution if LangGraph fails.
    """
    state = _default_rag_state(
        file_path=file_path,
        doc_name=doc_name,
        qwen_api_key=qwen_api_key,
        skip_images=skip_images,
        skip_tables=skip_tables,
    )

    try:
        from src.indexer import reset_rag_collections

        reset_rag_collections()
        graph = build_indexing_graph()
        if graph:
            state = graph.invoke(state)
        else:
            raise RuntimeError("Graph build failed")
    except Exception as e:
        print(f"[rag_pipeline] Graph indexing failed ({e}), running sequentially")
        state = node_parse(state)
        state = node_index_text(state)
        state = node_caption_images(state)
        state = node_process_tables(state)

    return {
        "text_count": state.get("text_count", 0),
        "image_count": state.get("image_count", 0),
        "table_count": state.get("table_count", 0),
        "error": state.get("error", ""),
        "text_blocks": len(state.get("text_blocks", [])),
        "image_paths": len(state.get("image_paths", [])),
        "tables": len(state.get("tables", [])),
    }


def index_bilibili_video(video_url: str, qwen_api_key: str) -> dict:
    """Extract a Bilibili video's metadata and subtitles, then index them as text."""
    try:
        from src.indexer import index_text, reset_rag_collections
        from src.tools.bilibili_tool import extract_bilibili_transcript

        transcript = extract_bilibili_transcript(video_url)
        reset_rag_collections()
        text_count = index_text(transcript.text_blocks, transcript.title, reset=False)

        return {
            "text_count": text_count,
            "image_count": 0,
            "table_count": 0,
            "subtitle_count": transcript.subtitle_count,
            "title": transcript.title,
            "owner": transcript.owner,
            "published": transcript.published,
            "duration": transcript.duration,
            "view_count": transcript.view_count,
            "source_url": transcript.url,
            "error": "",
        }
    except Exception as e:
        print(f"[rag_pipeline] Bilibili indexing failed: {e}\n{traceback.format_exc()}")
        return {
            "text_count": 0,
            "image_count": 0,
            "table_count": 0,
            "subtitle_count": 0,
            "title": "",
            "owner": "",
            "published": "",
            "duration": "",
            "view_count": "",
            "source_url": video_url,
            "error": str(e),
        }


def answer_query(query: str, qwen_api_key: str) -> dict:
    """
    Answer a query against indexed content.
    Returns answer + retrieved results + query types.
    """
    state = _default_rag_state(qwen_api_key=qwen_api_key)
    state["query"] = query

    try:
        graph = build_query_graph()
        if graph:
            state = graph.invoke(state)
        else:
            raise RuntimeError("Graph build failed")
    except Exception as e:
        print(f"[rag_pipeline] Query graph failed ({e}), running sequentially")
        state = node_route_query(state)
        state = node_retrieve(state)
        state = node_generate(state)

    return {
        "answer": state.get("answer", "No answer generated."),
        "retrieved_results": state.get("retrieved_results", []),
        "query_types": [qt.value if hasattr(qt, "value") else str(qt) for qt in state.get("query_types", [])],
        "error": state.get("error", ""),
    }
