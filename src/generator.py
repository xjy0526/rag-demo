"""
generator.py — Final answer generation using Qwen.
"""

from __future__ import annotations
from src.llm_clients import call_llm


def generate_answer(
    query: str,
    retrieved_results: list[dict],
    qwen_llm,
    fallback_llm=None,
) -> str:
    """
    Generate a grounded, source-cited answer from multimodal retrieved context.
    Uses Qwen for final generation.
    """
    if not retrieved_results:
        return (
            "当前没有检索到可用上下文，所以我还不能直接回答这个问题。\n\n"
            "你可以这样处理：\n"
            "1. 先确认已经成功索引 PDF 或哔哩哔哩视频。\n"
            "2. 把问题问得更具体一些，例如直接带上章节名、表格名、视频主题或术语。\n"
            "3. 如果你想问“具体有几步、应该怎么做”，最好把任务对象也说清楚，比如“这个实验有几步”或“视频里部署流程有几步”。\n\n"
            "本次使用的内容类型：无。"
        )

    # Separate by modality
    text_chunks, image_captions, table_descriptions = [], [], []
    image_refs = []

    for result in retrieved_results:
        modality = result.get("modality", "text")
        content = result.get("content", "").strip()
        if modality == "text":
            text_chunks.append(content)
        elif modality == "image":
            image_captions.append(content)
            path = result.get("metadata", {}).get("image_path", "")
            if path:
                image_refs.append(path)
        elif modality == "table":
            table_descriptions.append(content)

    if not any([text_chunks, image_captions, table_descriptions]):
        return (
            "已经执行了检索，但没有找到足够可回答问题的文本、图片或表格内容。\n\n"
            "建议你换一种更具体的问法，或者先重新索引当前资料后再试一次。\n\n"
            "本次使用的内容类型：无。"
        )

    text_section = "\n\n".join(text_chunks) or "No text context retrieved."
    image_section = "\n\n".join(image_captions) or "No image context retrieved."
    table_section = "\n\n".join(table_descriptions) or "No table context retrieved."

    prompt = f"""你是一个严谨的多模态研究助手。请严格只根据给定上下文回答问题。
如果上下文不足以支持结论，不要编造；请明确说明缺失了什么信息，并建议用户如何补充问题。
回答语言必须为中文。
请在答案中明确哪些内容来自 text / image / table。

=== TEXT CONTEXT ===
{text_section}

=== IMAGE DESCRIPTIONS ===
{image_section}

=== TABLE DATA ===
{table_section}

=== QUESTION ===
{query}

=== ANSWER ===
请给出清晰、结构化、自然的中文回答。最后单独写一行：本次使用的内容类型：..."""

    llm = qwen_llm if qwen_llm is not None else fallback_llm
    answer = call_llm(llm, prompt, fallback="暂时无法生成回答，请检查千问 API 配置。")

    # Append image references
    if image_refs:
        refs = "\n".join(f"📷 {p}" for p in image_refs)
        answer = f"{answer}\n\n参考图片：\n{refs}"

    return answer
