# -*- coding: utf-8 -*-
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

# 自动创建目录
for path in [DATA_DIR, IMAGE_SAVE_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)