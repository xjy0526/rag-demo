"""
research_agent.py — LangGraph-based research agent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from typing import Optional, TypedDict


class ResearchState(TypedDict):
    query: str
    topic: str
    priority_channel: Optional[str]
    priority_repo_url: Optional[str]
    priority_paper_url: Optional[str]
    max_papers: int
    max_books: int
    max_repos: int
    papers: list
    books: list
    repos: list
    websites: list
    videos: list
    summary: str
    error: str
    qwen_api_key: str
    github_token: str
    output_language: str


def _default_state(
    query: str,
    qwen_api_key: str = "",
    github_token: str = "",
    output_language: str = "zh",
    priority_channel: str = "",
    priority_repo_url: str = "",
    priority_paper_url: str = "",
    max_papers: int = 20,
    max_books: int = 10,
    max_repos: int = 20,
) -> ResearchState:
    return {
        "query": query,
        "topic": query,
        "priority_channel": priority_channel or None,
        "priority_repo_url": priority_repo_url or None,
        "priority_paper_url": priority_paper_url or None,
        "max_papers": max_papers,
        "max_books": max_books,
        "max_repos": max_repos,
        "papers": [],
        "books": [],
        "repos": [],
        "websites": [],
        "videos": [],
        "summary": "",
        "error": "",
        "qwen_api_key": qwen_api_key,
        "github_token": github_token,
        "output_language": output_language,
    }


def node_refine_query(state: ResearchState) -> ResearchState:
    """Use Qwen to extract a clean topic and ML-focused search terms."""
    try:
        from src.llm_clients import call_llm, get_qwen_llm

        llm = get_qwen_llm(state["qwen_api_key"], temperature=0.1, max_tokens=2048)
        if llm:
            prompt = (
                "Extract the core ML/AI/Data Science topic from this query. "
                "Return ONLY the refined topic (3-8 words), no explanation.\n\n"
                f"Query: {state['query']}"
            )
            refined = call_llm(llm, prompt, fallback=state["query"])
            refined = refined.strip().strip('"').strip("'")
            state["topic"] = refined if len(refined) > 3 else state["query"]
    except Exception as e:
        print(f"[research_agent] Query refinement failed: {e}")
        state["topic"] = state["query"]
    return state


def node_search_papers(state: ResearchState) -> ResearchState:
    """Search ArXiv + Semantic Scholar for research papers."""
    try:
        from src.tools.arxiv_tool import merge_and_rank_papers, search_arxiv, search_semantic_scholar

        query = state["topic"]
        priority = state.get("priority_paper_url")
        arxiv_papers = search_arxiv(query, max_results=state["max_papers"], priority_query=priority)
        ss_papers = search_semantic_scholar(query, max_results=10)
        state["papers"] = merge_and_rank_papers(arxiv_papers, ss_papers, max_results=state["max_papers"])
    except Exception as e:
        print(f"[research_agent] Paper search failed: {e}\n{traceback.format_exc()}")
        state["papers"] = []
    return state


def node_search_books(state: ResearchState) -> ResearchState:
    """Search for books."""
    try:
        from src.tools.book_tool import search_books

        state["books"] = search_books(state["topic"], max_results=state["max_books"])
    except Exception as e:
        print(f"[research_agent] Book search failed: {e}")
        state["books"] = []
    return state


def node_search_repos(state: ResearchState) -> ResearchState:
    """Search GitHub repositories."""
    try:
        from src.tools.github_tool import search_github

        state["repos"] = search_github(
            state["topic"],
            max_results=state["max_repos"],
            github_token=state.get("github_token") or None,
            priority_repo_url=state.get("priority_repo_url") or None,
        )
    except Exception as e:
        print(f"[research_agent] Repo search failed: {e}")
        state["repos"] = []
    return state


def node_search_websites(state: ResearchState) -> ResearchState:
    """Search for authoritative websites and resources."""
    try:
        from src.tools.website_tool import search_websites

        state["websites"] = search_websites(
            state["topic"],
            max_results=10,
            priority_url=state.get("priority_repo_url") or None,
        )
    except Exception as e:
        print(f"[research_agent] Website search failed: {e}")
        state["websites"] = []
    return state


def node_search_videos(state: ResearchState) -> ResearchState:
    """Search Bilibili for educational videos."""
    try:
        from src.tools.bilibili_tool import search_bilibili_videos

        state["videos"] = search_bilibili_videos(
            state["topic"],
            max_results=10,
            priority_channel=state.get("priority_channel") or None,
        )
    except Exception as e:
        print(f"[research_agent] Video search failed: {e}")
        state["videos"] = []
    return state


def node_generate_summary(state: ResearchState) -> ResearchState:
    """Use Qwen to generate a structured research summary."""
    try:
        from src.llm_clients import call_llm, get_qwen_llm
        is_english = state.get("output_language") == "en"

        papers_ctx = "\n".join(
            f"- {p.title} ({p.published}) by {', '.join(p.authors[:2])} — {p.abstract[:150]}"
            for p in state["papers"][:5]
        ) or ("No papers found." if is_english else "未找到论文。")

        books_ctx = "\n".join(
            f"- {b.title} by {', '.join(b.authors[:2])} ({'Free' if b.is_free else 'Paid'})"
            for b in state["books"][:5]
        ) or ("No books found." if is_english else "未找到图书。")

        repos_ctx = "\n".join(
            f"- {r.full_name} ⭐{r.stars} — {r.description[:100]}"
            for r in state["repos"][:5]
        ) or ("No repos found." if is_english else "未找到仓库。")

        videos_ctx = "\n".join(
            f"- {v.title} by {v.channel} ({v.view_count} 播放)"
            for v in state["videos"][:5]
        ) or ("No videos found." if is_english else "未找到视频。")

        if is_english:
            prompt = f"""You are an expert ML/AI research assistant.
Generate a concise, structured research overview in English for the topic: "{state['topic']}"

Based on these findings:

TOP PAPERS:
{papers_ctx}

TOP BOOKS:
{books_ctx}

TOP REPOS:
{repos_ctx}

TOP VIDEOS:
{videos_ctx}

Write a 3-4 paragraph summary covering:
1. What this topic is about and why it matters
2. Key research directions and landmark papers
3. Practical tools and implementations available
4. Recommended learning path for a data scientist/ML engineer

Be specific and cite the resources above where relevant.
Your output must be entirely in English."""
        else:
            prompt = f"""你是一名资深 AI / ML 研究助手。
请围绕主题“{state['topic']}”生成一份简洁、结构化、适合中文阅读的研究综述。

请基于以下检索结果：

论文：
{papers_ctx}

图书：
{books_ctx}

仓库：
{repos_ctx}

视频：
{videos_ctx}

请用 3-4 段中文内容覆盖：
1. 这个主题是什么，为什么重要
2. 关键研究方向与代表性论文
3. 现有工具、实现与工程落地
4. 适合数据科学家或机器学习工程师的学习路径

请尽量具体，并在合适处引用上面的资源。输出必须全部使用中文。"""

        llm = get_qwen_llm(state["qwen_api_key"])
        state["summary"] = call_llm(
            llm,
            prompt,
            fallback=(
                (
                    f"Research summary for '{state['topic']}': "
                    f"Found {len(state['papers'])} papers, {len(state['books'])} books, "
                    f"{len(state['repos'])} repositories, and {len(state['videos'])} videos."
                )
                if is_english
                else (
                    f"主题“{state['topic']}”的研究摘要："
                    f"共找到 {len(state['papers'])} 篇论文、{len(state['books'])} 本图书、"
                    f"{len(state['repos'])} 个仓库，以及 {len(state['videos'])} 个视频。"
                )
            ),
        )
    except Exception as e:
        print(f"[research_agent] Summary generation failed: {e}\n{traceback.format_exc()}")
        if state.get("output_language") == "en":
            state["summary"] = (
                f"Found {len(state['papers'])} papers, {len(state['books'])} books, "
                f"{len(state['repos'])} repos, and {len(state['videos'])} videos for '{state['topic']}'."
            )
        else:
            state["summary"] = (
                f"围绕“{state['topic']}”共找到 {len(state['papers'])} 篇论文、"
                f"{len(state['books'])} 本图书、{len(state['repos'])} 个仓库和 {len(state['videos'])} 个视频。"
            )
    return state


def _run_searches_parallel(state: ResearchState) -> ResearchState:
    """Run independent search providers concurrently after the topic is refined."""
    search_jobs = {
        "papers": node_search_papers,
        "books": node_search_books,
        "repos": node_search_repos,
        "websites": node_search_websites,
        "videos": node_search_videos,
    }
    with ThreadPoolExecutor(max_workers=len(search_jobs)) as executor:
        futures = {
            executor.submit(node, dict(state)): result_key
            for result_key, node in search_jobs.items()
        }
        for future in as_completed(futures):
            result_key = futures[future]
            try:
                branch_state = future.result()
                state[result_key] = branch_state.get(result_key, [])
            except Exception as e:
                print(f"[research_agent] Parallel search failed for {result_key}: {e}")
                state[result_key] = []
    return state


def build_research_graph():
    """Build and compile the LangGraph research workflow."""
    try:
        from langgraph.graph import END, StateGraph

        builder = StateGraph(ResearchState)
        builder.add_node("refine_query", node_refine_query)
        builder.add_node("search_papers", node_search_papers)
        builder.add_node("search_books", node_search_books)
        builder.add_node("search_repos", node_search_repos)
        builder.add_node("search_websites", node_search_websites)
        builder.add_node("search_videos", node_search_videos)
        builder.add_node("generate_summary", node_generate_summary)

        builder.set_entry_point("refine_query")
        builder.add_edge("refine_query", "search_papers")
        builder.add_edge("search_papers", "search_books")
        builder.add_edge("search_books", "search_repos")
        builder.add_edge("search_repos", "search_websites")
        builder.add_edge("search_websites", "search_videos")
        builder.add_edge("search_videos", "generate_summary")
        builder.add_edge("generate_summary", END)
        return builder.compile()
    except Exception as e:
        print(f"[research_agent] Graph build failed: {e}\n{traceback.format_exc()}")
        return None


def run_research(
    query: str,
    qwen_api_key: str = "",
    github_token: str = "",
    output_language: str = "zh",
    priority_channel: str = "",
    priority_repo_url: str = "",
    priority_paper_url: str = "",
    max_papers: int = 20,
    max_books: int = 10,
    max_repos: int = 20,
) -> ResearchState:
    """Run the full research pipeline for a query."""
    state = _default_state(
        query=query,
        qwen_api_key=qwen_api_key,
        github_token=github_token,
        output_language=output_language,
        priority_channel=priority_channel,
        priority_repo_url=priority_repo_url,
        priority_paper_url=priority_paper_url,
        max_papers=max_papers,
        max_books=max_books,
        max_repos=max_repos,
    )

    try:
        state = node_refine_query(state)
        state = _run_searches_parallel(state)
        state = node_generate_summary(state)
        return state
    except Exception as e:
        print(f"[research_agent] Parallel execution failed ({e}), running sequentially")
        state = node_refine_query(state)
        state = node_search_papers(state)
        state = node_search_books(state)
        state = node_search_repos(state)
        state = node_search_websites(state)
        state = node_search_videos(state)
        state = node_generate_summary(state)
        return state
