# Multi-Demo

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Qwen](https://img.shields.io/badge/LLM-Qwen-yellow?style=flat-square)

## 项目简介

`Multi-Demo` 是一个面向 AI / ML 学习、研究与知识整理场景的多模态演示项目。

它主要提供两类工作流：

1. 文档与视频问答
   上传 PDF，或粘贴一个哔哩哔哩视频链接，系统会先建立索引，再进行基于内容的问答。
2. 研究探索
   输入一个 AI / ML 主题，系统会联合检索论文、图书、GitHub 仓库、网站和哔哩哔哩视频，并生成摘要。

## 当前能力

### 1. PDF 多模态问答

- 解析 PDF 文本内容
- 抽取图片并调用大模型视觉模型生成图像描述
- 抽取表格并生成适合检索的自然语言摘要
- 将文本、图片、表格统一写入 ChromaDB 检索库

### 2. 哔哩哔哩视频分析

- 输入哔哩哔哩视频链接
- 自动提取视频标题、简介、标签等元信息
- 优先抓取字幕并切分为可检索内容
- 将字幕与元信息一起接入 RAG 问答链路
- 没有字幕时，仍可基于元信息做有限分析

### 3. AI 研究探索

- 论文：ArXiv + Semantic Scholar
- 图书：Open Library / Google Books 等公开来源
- 仓库：GitHub 检索与排序
- 网站：文档、课程、博客、工具资源
- 视频：哔哩哔哩主题相关内容
- 摘要：由大模型统一生成，可切换中文或英文输出

## 技术架构

### 模型层

- Qwen / DashScope
  - 查询改写
  - 路由判断
  - 图片理解
  - 表格描述
  - 最终回答与摘要生成

### 检索层

- ChromaDB：持久化向量库
- 本地嵌入兜底：受限环境下也能完成基本索引与检索

### 编排层

- LangGraph：PDF 索引与问答流程
- 并行检索：研究探索会并发检索论文、图书、GitHub、网站和哔哩哔哩视频，减少等待时间

### 前端

- Streamlit

## 项目结构

```text
Multi-Demo/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
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
| `GITHUB_TOKEN` | GitHub Token，可选，用于提升 GitHub 检索速率限制 |

## 快速开始

### 方式一：直接本地运行

```bash
git clone https://github.com/xjy0526/multi-demo.git Multi-Demo
cd Multi-Demo

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
```

启动后访问：

```text
http://127.0.0.1:8501
```

### 方式二：Docker 运行

```bash
docker build -t multi-demo .
docker run --rm -p 7860:7860 --env-file .env multi-demo
```

启动后访问：

```text
http://127.0.0.1:7860
```

## 使用说明

### 文档问答

1. 打开“文档与视频问答”
2. 上传 PDF 文件
3. 按需勾选是否跳过图片理解或表格处理
4. 点击“索引 PDF”
5. 索引完成后直接提问

### 哔哩哔哩视频问答

1. 打开“文档与视频问答”
2. 切换到“哔哩哔哩视频”
3. 粘贴视频链接
4. 点击“解析并索引视频”
5. 索引完成后继续提问

### 研究探索

1. 打开“研究探索”
2. 输入主题
3. 可选填写优先 UP 主、GitHub 仓库或论文线索
4. 选择摘要语言
5. 执行检索并查看结果

## 注意事项

- 哔哩哔哩视频能力依赖视频页面可访问，以及字幕信息可获取
- 没有字幕的视频仍可做概览，但细粒度问答效果会下降
- PDF 和哔哩哔哩视频可以先建立可检索索引；图片理解、AI 摘要和最终回答依赖千问 API Key
- 未配置千问 API Key 时，研究探索仍会执行基础检索，但只返回降级摘要
- 当前问答默认围绕“当前已索引来源”工作；切换新来源会重置当前对话上下文
- 若运行环境无法使用外部嵌入模型，项目会自动退回到本地兜底嵌入策略，并对中文文本做基础分词处理
