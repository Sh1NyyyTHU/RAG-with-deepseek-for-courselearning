# Courseware QA System — 项目上下文

## 项目概述
本地 RAG 课程课件知识库问答系统，面向高电压工程课程。基于 BGE-M3 Embedding + ChromaDB + DeepSeek API + Streamlit。

## 启动方式
- 双击 `启动.bat` 即可启动
- 或手动: `streamlit run app.py`
- 访问 http://localhost:8501

## 环境
- Conda 环境名: `courseware-qa`
- Conda 环境路径: `H:\conda_envs\courseware-qa`
- Conda 安装路径: `C:\ProgramData\miniconda3`
- Python 3.11, CUDA 12.6, RTX 4080
- DeepSeek API Key 在 `.env` 文件中（gitignored）
- HuggingFace 模型缓存: `~/.cache/huggingface/hub/models--BAAI--bge-m3`

## 关键决策与修复记录

### Embedding
- 使用 BAAI/bge-m3 via FlagEmbedding (BGEM3FlagModel)
- sentence-transformers 作为 fallback
- transformers 需要 >= 5.0.0（不然 XLMRobertaModel 不支持 dtype 参数）
- `启动.bat` 中设置了 `HF_HUB_OFFLINE=1` 跳过 HF 网络校验，因为国内直连 HuggingFace 经常超时

### Prompt 策略（已修改）
- 初版: 课件不足时模型拒绝回答 → 太死板
- 修改后: 课件不足时模型用自己的知识回答，但必须标注【课件依据】vs【模型知识】
- 严禁将模型知识伪装成课件内容、严禁编造页码

### ChromaDB
- 持久化目录: `data/chroma_db`
- 切换环境后如果报 HNSW index 错误，删除 `data/chroma_db` 重建
- 文件去重基于 SHA256

### Streamlit 流式输出
- 使用 `st.write_stream()` 处理流式 markdown，避免 chunk 渲染导致的格式错乱
- 用 list `[0]` 做 mutable closure 收集完整回答用于历史记录

### 启动.bat 演进
- 初版: 依赖 conda 在 PATH → 普通 cmd 找不到
- 修复: 自动搜索常见 conda 安装路径
- 不再用 conda activate/conda run，直接调用 env 内的 python.exe
- 全 ASCII 英文，避免中文乱码

## 文件结构
```
CoursewareQA/
├── app.py                # Streamlit 主应用
├── config.py             # 全局配置
├── 启动.bat               # Windows 一键启动
├── CLAUDE.md             # 本文件
├── .env                  # API Key (gitignored)
├── .env.example          # 环境变量模板
├── src/
│   ├── pdf_parser.py     # PDF 解析 (PyMuPDF)
│   ├── text_splitter.py  # 文本切分 (800/120)
│   ├── embedding_service.py # Embedding (BGE-M3)
│   ├── vector_store.py   # ChromaDB
│   ├── retriever.py      # 检索 (含 post_processor 接口)
│   ├── deepseek_client.py # DeepSeek API
│   ├── prompt_templates.py # 三种模式 Prompt
│   ├── knowledge_base.py # 知识库编排
│   ├── history_manager.py # 历史记录 + Markdown 导出
│   ├── ocr_adapter.py    # OCR 适配器 (PaddleOCR 可选)
│   └── utils.py          # 工具函数
├── tests/                # 21 个单元测试
├── scripts/              # 辅助脚本
└── data/                 # PDF, chroma_db, exports, logs
```

## 当前状态
- 21/21 单元测试通过
- 三种问答模式: QA / 知识点讲解 / 做题模式
- 做题模式启用 reasoning_effort="high"
- PDF 检索返回正确页码 (已验证 smoke test)
- 持久化索引正常工作
- OCR 接口预留但未启用
- Reranker 接口预留但未启用
