"""
tools/bilibili_tool.py — Bilibili search and transcript extraction helpers.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, quote, urlparse

import requests


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


@dataclass
class BilibiliVideo:
    title: str
    channel: str
    url: str
    description: str
    view_count: str
    duration: str
    published: str
    thumbnail: str
    is_priority_channel: bool = False


@dataclass
class BilibiliTranscript:
    title: str
    url: str
    owner: str
    description: str
    published: str
    duration: str
    view_count: str
    thumbnail: str
    tags: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    subtitle_count: int = 0


def search_bilibili_videos(
    query: str,
    max_results: int = 10,
    priority_channel: Optional[str] = None,
) -> list[BilibiliVideo]:
    """Search Bilibili educational videos."""
    videos: list[BilibiliVideo] = []
    try:
        response = requests.get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={
                "search_type": "video",
                "keyword": f"{query} 机器学习 教程",
                "page": 1,
            },
            headers=HEADERS,
            timeout=15,
        )
        data = response.json().get("data", {})
        for item in data.get("result", [])[: max(max_results * 2, 10)]:
            bvid = item.get("bvid")
            if not bvid:
                continue
            author = _clean_text(item.get("author") or item.get("upic_str") or "未知 UP 主")
            video = BilibiliVideo(
                title=_clean_text(item.get("title", "未知标题")),
                channel=author,
                url=f"https://www.bilibili.com/video/{bvid}",
                description=_clean_text(item.get("description", ""))[:200],
                view_count=_format_count(item.get("play")),
                duration=item.get("duration", "N/A"),
                published=_format_date(item.get("pubdate")),
                thumbnail=_normalize_url(item.get("pic", "")),
            )
            if priority_channel and priority_channel.lower() in author.lower():
                video.is_priority_channel = True
            videos.append(video)
    except Exception as e:
        print(f"[bilibili_tool] Search failed: {e}")

    if not videos:
        videos.append(
            BilibiliVideo(
                title=f"前往哔哩哔哩搜索：{query}",
                channel="Bilibili Search",
                url=f"https://search.bilibili.com/all?keyword={quote(query)}",
                description="未能获取结构化结果，可直接跳转到 B 站搜索页。",
                view_count="N/A",
                duration="N/A",
                published="N/A",
                thumbnail="",
            )
        )

    videos.sort(key=lambda v: (v.is_priority_channel, _sort_views(v.view_count)), reverse=True)
    return videos[:max_results]


def extract_bilibili_transcript(video_url: str) -> BilibiliTranscript:
    """Extract metadata and subtitle text from a Bilibili video page."""
    video_url = normalize_bilibili_url(video_url)
    page_response = requests.get(video_url, headers=HEADERS, timeout=20, allow_redirects=True)
    page_response.raise_for_status()
    resolved_url = page_response.url
    page_html = page_response.text

    page_no = _extract_page_number(resolved_url)
    bvid = _extract_bvid(resolved_url) or _extract_bvid(page_html)
    if not bvid:
        raise ValueError("无法从链接中识别 B 站视频 BV 号。")

    video_info = _fetch_video_info(bvid)
    video_data = video_info.get("data") or {}
    pages = video_data.get("pages") or []
    page_idx = min(max(page_no - 1, 0), max(len(pages) - 1, 0))
    page_meta = pages[page_idx] if pages else {}
    cid = str(page_meta.get("cid") or video_data.get("cid") or "")
    if not cid:
        raise ValueError("无法获取视频分 P 的 cid。")

    playinfo = _extract_embedded_json(page_html, "window.__playinfo__=")
    subtitle_candidates = _extract_subtitle_candidates(playinfo)
    if not subtitle_candidates:
        subtitle_candidates = _fetch_subtitle_candidates_via_api(bvid=bvid, cid=cid)

    subtitle_lines = _download_best_subtitle(subtitle_candidates)
    tags = _fetch_tags(bvid)

    title = page_meta.get("part") or video_data.get("title") or bvid
    description = (video_data.get("desc") or "").strip()
    owner = (video_data.get("owner") or {}).get("name", "未知 UP 主")
    published = _format_date(video_data.get("pubdate"))
    duration = _format_duration(page_meta.get("duration") or video_data.get("duration"))
    view_count = _format_count((video_data.get("stat") or {}).get("view"))
    thumbnail = _normalize_url(video_data.get("pic", ""))

    text_blocks = _build_text_blocks(
        title=title,
        owner=owner,
        published=published,
        duration=duration,
        view_count=view_count,
        description=description,
        tags=tags,
        subtitle_lines=subtitle_lines,
    )

    return BilibiliTranscript(
        title=title,
        url=resolved_url,
        owner=owner,
        description=description,
        published=published,
        duration=duration,
        view_count=view_count,
        thumbnail=thumbnail,
        tags=tags,
        text_blocks=text_blocks,
        subtitle_count=len(subtitle_lines),
    )


def normalize_bilibili_url(raw_text: str) -> str:
    """Extract a usable Bilibili URL from arbitrary pasted text."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("请输入哔哩哔哩视频链接。")

    text = text.replace("，", ",").replace("。", ".").replace("【", " ").replace("】", " ")
    match = re.search(r"https?://[^\s]+", text)
    if match:
        candidate = match.group(0).strip("()[]{}<>.,!?\"'")
    else:
        match = re.search(r"(www\.(?:bilibili\.com|b23\.tv)/[^\s]+)", text, flags=re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip("()[]{}<>.,!?\"'")
            candidate = f"https://{cleaned}"
        else:
            raise ValueError("未能从输入内容中识别出有效的哔哩哔哩链接。")

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if not any(domain in host for domain in ("bilibili.com", "b23.tv")):
        raise ValueError("识别到的链接不是哔哩哔哩链接，请重新粘贴。")
    return candidate


def _fetch_video_info(bvid: str) -> dict:
    response = requests.get(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _fetch_tags(bvid: str) -> list[str]:
    try:
        response = requests.get(
            "https://api.bilibili.com/x/tag/archive/tags",
            params={"bvid": bvid},
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        return [item.get("tag_name", "") for item in response.json().get("data", []) if item.get("tag_name")]
    except Exception:
        return []


def _extract_subtitle_candidates(playinfo: dict) -> list[dict]:
    subtitles = (((playinfo or {}).get("data") or {}).get("subtitle") or {}).get("subtitles") or []
    results = []
    for item in subtitles:
        url = item.get("subtitle_url")
        if url:
            results.append(
                {
                    "lang": item.get("lan") or item.get("lan_doc") or "",
                    "url": _normalize_url(url),
                }
            )
    return results


def _fetch_subtitle_candidates_via_api(bvid: str, cid: str) -> list[dict]:
    try:
        response = requests.get(
            "https://api.bilibili.com/x/player/wbi/v2",
            params={"bvid": bvid, "cid": cid},
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json().get("data", {})
        subtitles = (data.get("subtitle") or {}).get("subtitles") or []
        return [
            {
                "lang": item.get("lan") or item.get("lan_doc") or "",
                "url": _normalize_url(item.get("subtitle_url", "")),
            }
            for item in subtitles
            if item.get("subtitle_url")
        ]
    except Exception as e:
        print(f"[bilibili_tool] Subtitle metadata fetch failed: {e}")
        return []


def _download_best_subtitle(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    def rank(item: dict) -> tuple[int, str]:
        lang = (item.get("lang") or "").lower()
        preferred = ["zh-cn", "zh-hans", "ai-zh", "zh", "en"]
        score = preferred.index(lang) if lang in preferred else len(preferred)
        return (score, lang)

    for candidate in sorted(candidates, key=rank):
        try:
            subtitle_url = candidate.get("url")
            if not subtitle_url:
                continue
            response = requests.get(subtitle_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            body = response.json().get("body") or []
            if body:
                return body
        except Exception as e:
            print(f"[bilibili_tool] Subtitle download failed: {e}")
    return []


def _build_text_blocks(
    title: str,
    owner: str,
    published: str,
    duration: str,
    view_count: str,
    description: str,
    tags: list[str],
    subtitle_lines: list[dict],
) -> list[str]:
    header = [
        f"视频标题：{title}",
        f"UP主：{owner}",
        f"发布时间：{published}",
        f"时长：{duration}",
        f"播放量：{view_count}",
    ]
    if tags:
        header.append(f"标签：{'、'.join(tags[:10])}")
    if description:
        header.append(f"简介：{description}")

    blocks = ["\n".join(header)]
    if not subtitle_lines:
        blocks.append(
            "未找到可用字幕。你仍可以基于视频标题、简介和标签进行概览分析，但细粒度内容问答会受限。"
        )
        return blocks

    current_lines: list[str] = []
    current_len = 0
    for item in subtitle_lines:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        start = _format_timestamp(item.get("from", 0))
        end = _format_timestamp(item.get("to", 0))
        line = f"[{start}-{end}] {content}"
        current_lines.append(line)
        current_len += len(line)
        if current_len >= 1200:
            blocks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


def _extract_embedded_json(page_html: str, marker: str) -> dict:
    idx = page_html.find(marker)
    if idx == -1:
        return {}
    start = idx + len(marker)
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(page_html[start:])
        return data
    except Exception:
        return {}


def _extract_page_number(url: str) -> int:
    query = parse_qs(urlparse(url).query)
    try:
        return max(int(query.get("p", ["1"])[0]), 1)
    except Exception:
        return 1


def _extract_bvid(text: str) -> str:
    match = re.search(r"(BV[0-9A-Za-z]{10})", text)
    return match.group(1) if match else ""


def _clean_text(value: str) -> str:
    stripped = re.sub(r"<[^>]+>", "", value or "")
    return html.unescape(stripped).replace("\n", " ").strip()


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    return url


def _format_count(value) -> str:
    try:
        num = int(float(value))
    except Exception:
        return str(value or "N/A")

    if num >= 100000000:
        return f"{num / 100000000:.1f}亿"
    if num >= 10000:
        return f"{num / 10000:.1f}万"
    return f"{num}"


def _sort_views(view_count: str) -> int:
    if view_count.endswith("亿"):
        return int(float(view_count[:-1]) * 100000000)
    if view_count.endswith("万"):
        return int(float(view_count[:-1]) * 10000)
    try:
        return int(view_count)
    except Exception:
        return 0


def _format_date(timestamp) -> str:
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def _format_duration(seconds) -> str:
    try:
        total = int(seconds)
    except Exception:
        return str(seconds or "N/A")

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_timestamp(seconds) -> str:
    try:
        total = int(float(seconds))
    except Exception:
        total = 0
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
