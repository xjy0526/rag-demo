# 多模态 RAG 研究助手

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Qwen](https://img.shields.io/badge/LLM-Qwen-yellow?style=flat-square)

## 项目简介

这是一个面向研究与知识整理场景的多模态 RAG 应用，现已完成两项核心改造：

1. 项目统一切换为 **千问 API（DashScope 兼容模式）**
2. 视频能力从 **YouTube** 改为 **哔哩哔哩**

当前项目支持两类核心工作流：

1. **文档 / 视频问答**
   你可以上传 PDF，或粘贴一个哔哩哔哩视频链接，系统会抽取内容建立索引，然后基于内容进行问答。
2. **研究探索**
   围绕某个 AI / ML / Data Science 主题，同时检索论文、图书、GitHub 仓库、网站与哔哩哔哩视频，并生成摘要。

## 当前功能

### 1. PDF 多模态问答

- 解析 PDF 文本
- 抽取图片并调用千问 VL 生成图像描述
- 抽取表格并调用千问生成适合检索的自然语言描述
- 使用 ChromaDB 建立文本 / 图片 / 表格统一检索

### 2. 哔哩哔哩链接视频分析

- 输入 Bilibili 视频链接
- 自动抓取视频元信息
- 优先提取视频字幕并切分为可检索文本块
- 将字幕与简介一起接入现有 RAG 问答链路
- 若视频无可用字幕，仍会基于标题、简介、标签提供有限分析

### 3. 研究探索

- 论文：ArXiv + Semantic Scholar
- 图书：免费资源优先，并补充 Open Library / Google Books
- 代码仓库：GitHub 检索与排序
- 网站：文档、课程、博客、工具
- 视频：哔哩哔哩教学视频检索
- 摘要：由千问统一生成研究概览

## 技术架构

### 模型层

- **Qwen / DashScope**
  - 查询改写
  - 问题路由
  - 表格描述
  - 图片理解
  - 最终答案生成

### 检索层

- **ChromaDB**：持久化向量库
- **Sentence Transformers**：`all-MiniLM-L6-v2`

### 编排层

- **LangGraph**
  - PDF 索引流程
  - 问答流程
  - 研究探索流程

### 前端

- **Streamlit**

## 项目结构

```text
multimodal-rag-research-assistant/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
└── src/
    ├── chroma_store.py
    ├── config.py
    ├── generator.py
    ├── indexer.py
    ├── llm_clients.py
    ├── multimodal_parser.py
    ├── query_router.py
    ├── rag_pipeline.py
    ├── research_agent.py
    ├── retriever.py
    ├── ui_components.py
    └── tools/
        ├── arxiv_tool.py
        ├── bilibili_tool.py
        ├── book_tool.py
        ├── github_tool.py
        └── website_tool.py
```

## 环境变量

复制 `.env.example` 后填写：

| 变量名 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 千问 / DashScope API Key |
| `QWEN_TEXT_MODEL` | 文本模型，默认 `qwen-plus` |
| `QWEN_VISION_MODEL` | 视觉模型，默认 `qwen3-vl-flash-2026-01-22` |
| `GITHUB_TOKEN` | GitHub Token，可选 |

## 快速开始

```bash
git clone https://github.com/Ashutosh-AIBOT/multimodal-rag-research-assistant.git
cd multimodal-rag-research-assistant

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
```

启动后打开 `http://localhost:8501`。

## 使用说明

### 文档问答

1. 打开“文档与视频问答”
2. 上传 PDF
3. 点击“索引 PDF”
4. 索引完成后直接开始提问

### 哔哩哔哩视频问答

1. 打开“文档与视频问答”
2. 切到“哔哩哔哩视频”
3. 粘贴视频链接
4. 点击“解析并索引视频”
5. 索引完成后直接提问

### 研究探索

1. 打开“研究探索”
2. 输入主题
3. 可选填写优先 UP 主 / GitHub 仓库 / 论文线索
4. 执行检索并查看摘要与分类结果

## 注意事项

- 哔哩哔哩视频分析依赖视频页面可访问，以及字幕信息可获取
- 若视频没有字幕，系统仍可基于元信息做概览，但细粒度问答效果会下降
- 图片理解与最终回答都依赖千问 API Key
- 当前索引设计默认围绕“当前单一来源”进行问答；重新索引新来源会清空旧索引

