# -*- coding: utf-8 -*-
import re
import os
from pathlib import Path

# =============================================================================
# 1. 基础与目录路径配置
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SOURCE_PDF_FILE = DATA_DIR / "Tesla_Manual.pdf"
IMAGE_SAVE_DIR = DATA_DIR / "extracted_images"

# =============================================================================
# 2. 核心模型路径配置
# =============================================================================
# M3E-Small 语义模型的本地绝对路径
EMBEDDING_M3E_SMALL_DIR = str(MODELS_DIR / "AI-ModelScope/m3e-small")

# =============================================================================
# 3. 文本分块 (Chunking) 与微服务配置
# =============================================================================
# [原生路线] 传统规则分块参数
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# [语义路线] M3E 语义微服务参数
CHUNKING_BASE_URL = 'http://localhost:6000/v1/semantic-chunks'
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20

# =============================================================================
# 4. 文本清洗配置 (Text Cleaning Rules)
# =============================================================================
# 清洗后的干净数据暂存路径 (来自你的项目评估报告架构)
CLEANED_DATA_DIR = DATA_DIR / "processed_docs"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 魔法 1：狙击页眉页脚的正则表达式
# 匹配规则：
# 1. ^\s*\d+\s*$ -> 匹配只有一串数字的行（通常是页码）
# 2. .*Tesla.*用户手册.* -> 匹配包含这些关键字的无用标题行
CLEAN_HEADER_FOOTER_PATTERN = re.compile(r'(^\s*\d+\s*$)|(.*用户手册.*)|(.*Model 3.*)', re.IGNORECASE | re.MULTILINE)

# 魔法 2：并发清洗的最大线程数 (未来如果你要洗 100 本手册时会用到)
CLEAN_MAX_WORKERS = 2

# =============================================================================
# 5. LLM 大模型配置 (指向本地 vLLM 服务)
# =============================================================================
VLLM_MODEL_NAME = ""
VLLM_BASE_URL = ""
VLLM_API_KEY = ""
# 因为我们要在本地用 vLLM 启动大模型，默认端口是 8000
LLM_API_URL = "http://127.0.0.1:8000/v1"  
# 本地模型不需要真实的 API Key，随便填一个就行
LLM_API_KEY = "sk-local-token"
# 这里写你接下来准备下载/使用的 Qwen3.5 模型名称
LLM_MODEL_NAME = "Qwen3.5-9B" # 提示：如果你用的 Qwen3.5，名称根据你实际启动的参数来定

# 6.Milvus 数据库文件存放路径
MILVUS_DB_FILE = str(DATA_DIR / "saved_index" / "tesla_milvus.db")
MILVUS_COLLECTION_NAME = "tesla_bge_m3_hybrid"

# 7.BGE-M3 模型路径和参数 (填入你在 AutoDL 上的实际下载路径，或者直接用 HuggingFace 的仓库名)
EMBEDDING_BGE_M3_DIR = str(MODELS_DIR / "bge-m3")
EMBEDDING_BATCH_SIZE = 50      # Embedding 批次大小

# =============================================================================
# 8. MongoDB 数据库配置
# =============================================================================
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'tesla_rag_db')

# =============================================================================
# 9. 清洗好后的数据路径配置
# =============================================================================
SPLIT_DOCS_FILE = str(DATA_DIR / "processed/split_docs.pkl")

# =============================================================================
# 10. bge-reranker-v2-m3重排序模型路径
RERANK_BGE_M3_DIR = str(MODELS_DIR / "bge-reranker-v2-m3")

# =============================================================================

# 11. LLM 重试配置
LLM_RETRY_ATTEMPTS = 3          # 重试次数
LLM_RETRY_WAIT_SECONDS = 2      # 重试间隔(秒)


MILVUS_ID_MAX_LENGTH = 128     # ID 字段最大长度

# 自动创建目录
for path in [DATA_DIR, IMAGE_SAVE_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)