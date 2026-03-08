# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
import logging
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从配置中心拉取所有需要的参数！
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_BASE_URL, 
    HTTP_POOL_CONNECTIONS, HTTP_POOL_MAXSIZE
)

logger = logging.getLogger(__name__)

class ChunkingClient:
    """高度工程化的双引擎分块服务客户端"""

    def __init__(self, strategy="rule"):
        """
        :param strategy: 可选 "rule" (原生文本分割) 或 "semantic" (API语义分割)
        """
        self.strategy = strategy.lower()
        
        if self.strategy == "rule":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
            )
            print(f"🔧 启用 [原生规则分割] (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
            
        elif self.strategy == "semantic":
            self.base_url = CHUNKING_BASE_URL
            self.session = self._init_session()
            print(f"🧠 启用 [M3E语义分割服务] (Target URL: {self.base_url})")
            
        else:
            raise ValueError("❌ 不支持的分块策略，请选择 'rule' 或 'semantic'")

    def _init_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504], allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def chunk_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if self.strategy == "rule":
            return self.text_splitter.split_text(text)
        elif self.strategy == "semantic":
            return self._call_semantic_api(text)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(requests.RequestException))
    def _call_semantic_api(self, sentences: str) -> List[str]:
        payload = {"sentences": sentences, "group_size": 10}
        headers = {"Content-Type": "application/json"}
        try:
            response = self.session.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json().get("chunks", [])
        except Exception as e:
            logger.error(f"Semantic chunk request failed: {e}")
            raise e

# ================= 测试代码 =================
if __name__ == '__main__':
    test_text = "特斯拉Model 3非常智能。当您坐在驾驶座上时，踩下制动踏板即可启动。\n\n另外，空调系统的滤芯需要每年更换一次。以保证车内空气的清新。"
    
    print("\n--- ✂️ 测试 1: 调用原生规则分割 ---")
    rule_client = ChunkingClient(strategy="rule")
    for i, c in enumerate(rule_client.chunk_text(test_text)):
        print(f"[{i+1}] {c}")
        
    print("\n--- 🧠 测试 2: 调用 M3E 语义分割微服务 ---")
    semantic_client = ChunkingClient(strategy="semantic")
    try:
        for i, c in enumerate(semantic_client.chunk_text(test_text)):
            print(f"[{i+1}] {c}")
    except Exception as e:
        print(f"⚠️ 语义分割失败！请确保你在另一个终端运行了 `uv run python services/api_server.py`\n报错详情: {e}")