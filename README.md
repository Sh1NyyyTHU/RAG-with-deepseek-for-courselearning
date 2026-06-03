# 课程课件智能问答与做题助手

基于 RAG（检索增强生成）的本地课程课件知识库问答系统，专为高电压工程课程设计。

## 功能特性

- 📄 **PDF 课件解析**：逐页解析 PDF，保留精确页码
- 🔍 **向量检索**：基于 BGE-M3 本地 Embedding + ChromaDB 向量数据库
- 💬 **三种回答模式**：
  - **课件问答**：基于课件内容回答问题
  - **知识点讲解**：以教师视角系统讲解知识点
  - **做题模式**：逐步推导解题，区分课件依据与模型推导
- 📖 **来源追踪**：每个回答都引用具体的 PDF 文件名和页码
- 🔄 **增量更新**：支持新增、删除、重新索引 PDF
- 💾 **持久化存储**：重启应用后索引数据不丢失
- 🖼️ **OCR 支持**（可选）：上传题目图片自动识别文字
- 🎨 **Streamlit Web UI**：清晰的三栏布局界面

## 系统要求

- Windows 10/11 或 Linux/macOS
- Python 3.11+（推荐 3.13）
- NVIDIA GPU（可选，CPU 也可运行）
- DeepSeek API Key

## 快速启动（Windows）

### 1. 获取代码

```bash
cd CoursewareQA
```

### 2. 创建虚拟环境（推荐使用 Conda）

```bash
# 方式 A：用 environment.yml 一键创建
conda env create -f environment.yml
conda activate courseware-qa
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 方式 B：手动创建
conda create -n courseware-qa python=3.11 -y
conda activate courseware-qa
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 如果不需要 GPU，可跳过 CUDA 版 torch 安装，直接用：
# pip install torch torchvision torchaudio
```

### 3. 安装依赖

如果使用方式 A 或 B 已按上一步操作，依赖已安装完毕。

```bash
# 仅当跳过上面步骤时：
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4. 配置 API Key

```bash
# 复制环境变量模板
copy .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

或者通过系统环境变量设置：

```bash
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
# PowerShell: $env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 5. 运行环境检查

```bash
python env_setup_check.py
```

### 6. 启动应用

```bash
streamlit run app.py
```

打开浏览器访问：http://localhost:8501

### 7. 测试 API 连通性（可选）

```bash
python scripts/test_deepseek_api.py
```

## 项目结构

```
CoursewareQA/
├── app.py                  # Streamlit 主应用
├── config.py               # 全局配置
├── env_setup_check.py      # 环境检查脚本
├── requirements.txt        # Python 依赖
├── environment.yml         # Conda 环境配置
├── .env.example            # 环境变量模板
├── .gitignore
├── README.md
├── data/                   # 数据目录
│   ├── pdfs/               # 上传的 PDF 文件
│   ├── chroma_db/          # 向量数据库持久化
│   ├── exports/            # 问答记录导出
│   └── logs/               # 日志文件
├── src/                    # 核心源码
│   ├── pdf_parser.py       # PDF 解析（PyMuPDF）
│   ├── text_splitter.py    # 文本切分
│   ├── embedding_service.py # Embedding 服务（BGE-M3）
│   ├── vector_store.py     # 向量数据库（ChromaDB）
│   ├── retriever.py        # 检索器
│   ├── deepseek_client.py  # DeepSeek API 客户端
│   ├── prompt_templates.py # Prompt 模板
│   ├── knowledge_base.py   # 知识库编排层
│   ├── history_manager.py  # 历史记录管理
│   ├── ocr_adapter.py      # OCR 适配器
│   └── utils.py            # 工具函数
├── scripts/                # 辅助脚本
│   ├── ingest_folder.py    # 批量导入 PDF
│   ├── test_deepseek_api.py # API 连通性测试
│   └── start_app.bat       # Windows 启动脚本
└── tests/                  # 单元测试
    ├── test_pdf_parser.py
    ├── test_text_splitter.py
    ├── test_vector_store.py
    └── test_retriever.py
```

## 使用说明

### 知识库管理（Tab 1）

1. 在左侧边栏或 Tab 1 中上传 PDF 课件
2. 点击"建立索引"按钮
3. 查看已索引文件列表及 chunk 数量
4. 支持删除、重新索引、清空知识库

### 问答与做题（Tab 2）

1. 在左侧边栏选择回答模式（课件问答/知识点讲解/做题模式）
2. 输入问题或题目
3. 可选：上传题目截图（需要 PaddleOCR）
4. 点击"提交"
5. 查看流式回答，展开查看引用来源和检索片段

### 批量索引已有 PDF

```bash
# 将所有 PDF 放到 data/pdfs/ 目录下，然后运行：
python scripts/ingest_folder.py

# 或指定其他目录：
python scripts/ingest_folder.py "E:\课件\高电压工程"
```

## 配置说明

所有配置通过环境变量或 `.env` 文件管理：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | - |
| `DEEPSEEK_MODEL` | DeepSeek 模型名称 | `deepseek-v4-pro` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `EMBEDDING_MODEL_DIR` | Embedding 模型目录 | 自动下载 |

配置文件 `config.py` 中的可调参数：
- `CHUNK_SIZE`: 文本切分大小（默认 800 字符）
- `CHUNK_OVERLAP`: 切分重叠（默认 120 字符）
- `DEFAULT_TOP_K`: 检索返回数量（默认 8）
- `DEEPSEEK_TIMEOUT`: API 超时（默认 120 秒）

## 技术架构

```
用户问题 → Embedding → ChromaDB 检索 → 相关片段
                                            ↓
                                  构建 Prompt（含上下文）
                                            ↓
                                   DeepSeek API 生成
                                            ↓
                                    流式输出回答
                                    （含来源引用）
```

- **Embedding**：BAAI/bge-m3（本地运行，CUDA 加速）
- **向量数据库**：ChromaDB（持久化到本地磁盘）
- **LLM**：DeepSeek API（OpenAI 兼容接口）
- **PDF 解析**：PyMuPDF（逐页保留页码）
- **前端**：Streamlit

## 运行测试

```bash
python -m pytest tests/ -v
```

## 验收状态

- [x] 可通过 `streamlit run app.py` 启动
- [x] 可上传文字型 PDF 并建立索引
- [x] 持久化向量索引，重启后仍存在
- [x] 检索返回正确的文件名和页码
- [x] 三种回答模式（QA/教学/做题）
- [x] 做题模式分步骤输出
- [x] 知识不足时不编造答案
- [x] API Key 不硬编码
- [x] CPU 模式可运行
- [x] GPU 可用时自动使用 CUDA
- [x] 单元测试全部通过（21/21）

## 可选增强（未实现）

- [ ] Reranker 接入（接口已预留）
- [ ] PaddleOCR 扫描件识别（接口已预留）
- [ ] 多轮对话历史
- [ ] 用户认证
- [ ] Docker 部署
- [ ] 双语支持（中英文自动切换）

## 常见问题

**Q: 启动时 Embedding 模型加载失败？**
A: 检查网络连接。首次运行会从 HuggingFace 下载 BGE-M3 模型（约 2GB）。如果网络受限，可手动下载模型并设置 `EMBEDDING_MODEL_DIR` 环境变量。

**Q: API 调用失败？**
A: 检查 `.env` 文件中的 `DEEPSEEK_API_KEY` 是否正确。可在侧边栏点击"测试 API 连通性"。

**Q: 检索结果不相关？**
A: 尝试调整侧边栏中的 `top_k` 参数，或提高 `chunk_size`。

**Q: CUDA 内存不足？**
A: Embedding 模型会自动回退到 CPU。可以在 `config.py` 中手动设置 `EMBEDDING_DEVICE = "cpu"`。

## License

MIT
