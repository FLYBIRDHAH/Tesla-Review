# -*- coding: utf-8 -*-
"""LLM 客户端单例，复用连接池"""
import logging
from typing import Optional
from openai import OpenAI
from httpx import Timeout
from config.settings import LLM_API_URL, LLM_API_KEY

logger = logging.getLogger(__name__)

class SharedLLMClient:
    """单例模式的 LLM 客户端"""
    _instance: Optional["SharedLLMClient"] = None
    _client: Optional[OpenAI] = None

    def __new__(cls, *args, **kwargs):
        # 实现单例模式：第一次调用时创建实例，后续返回同一个实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # 避免重复初始化
        if SharedLLMClient._client is not None:
            return

        api_key = api_key or LLM_API_KEY or "EMPTY"
        base_url = base_url or LLM_API_URL

        # 配置 httpx 超时时间，保证大并发时不会轻易断开
        timeout = Timeout(connect=30.0, read=120.0, write=30.0, pool=120.0)

        SharedLLMClient._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=3
        )
        logger.info(f"Shared LLM client 初始化成功 (base_url={base_url})")

def get_shared_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """对外暴露的获取客户端的便捷函数"""
    if SharedLLMClient._instance is None:
        SharedLLMClient(api_key=api_key, base_url=base_url)
    return SharedLLMClient._client