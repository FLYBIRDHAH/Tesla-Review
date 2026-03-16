# -*- coding: utf-8 -*-
import os
import sys
import logging
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 从你的全局配置导入各种环境的 URL 和 Key
from config.settings import (
    LLM_MODEL_NAME, VLLM_MODEL_NAME,
    LLM_API_URL, LLM_API_KEY,
    VLLM_BASE_URL, VLLM_API_KEY
)
# 导入你写好的单例连接池
from src.llm.shared_client import get_shared_client
from config.prompts import Prompts

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, use_vllm=False, model_name=None):
        # 根据 use_vllm 参数，极其优雅地切换你的远程/本地大模型环境
        if use_vllm:
            api_key = VLLM_API_KEY
            base_url = VLLM_BASE_URL
            self.model_name = model_name or VLLM_MODEL_NAME
        else:
            api_key = LLM_API_KEY
            base_url = LLM_API_URL
            self.model_name = model_name or LLM_MODEL_NAME
            
        # 把对应的环境参数喂给你的单例工厂
        self.client = get_shared_client(api_key=api_key, base_url=base_url)
        logger.info(f"🧠 LLM Client 初始化完成，使用模型: {self.model_name} (vLLM模式: {use_vllm})")

    def chat(self, query: str, context: str, stream: bool = False):
        """统一调用入口"""
        if stream:
            return self._stream_chat(query, context)
        else:
            return self._non_stream_chat(query, context)

    def _build_kwargs(self, query: str, context: str, stream: bool):
        """封装通用请求参数"""
        prompt = Prompts.CHAT.format(context=context, query=query)
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个严谨的特斯拉问答助手。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.01,
            "top_p": 0.95,
            "frequency_penalty": 1.1,
            "stream": stream,
            "extra_body": {
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }

    def _non_stream_chat(self, query: str, context: str) -> str:
        """非流式：一次性吐出完整答案"""
        try:
            kwargs = self._build_kwargs(query, context, stream=False)
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ LLM 非流式生成失败: {e}")
            raise e

    def _stream_chat(self, query: str, context: str):
        """流式：打字机效果，提升用户体验"""
        try:
            kwargs = self._build_kwargs(query, context, stream=True)
            response = self.client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"❌ LLM 流式生成失败: {e}")
            raise e

# ================= 单元测试 =================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- 🚀 启动 LLM Client 测试 ---")
    
    try:
        # 默认关闭 vLLM，使用远程 API 测试更稳定
        llm = LLMClient(use_vllm=False)
        
        test_query = "方向盘怎么加热？"
        test_context = "1. 在触摸屏上点击控制图标。\n2. 点击方向盘加热图标即可开启。"
        
        print(f"\n🙋 用户提问: {test_query}")
        print(f"📚 检索上下文:\n{test_context}\n")
        
        print("⏳ [测试 1] 非流式输出测试...")
        full_res = llm.chat(test_query, test_context, stream=False)
        print(f"🤖 完整回答:\n{full_res}\n")
        
        print("⏳ [测试 2] 流式输出 (打字机效果) 测试...")
        print("🤖 实时回答: ", end="")
        for chunk in llm.chat(test_query, test_context, stream=True):
            print(chunk, end="", flush=True)
        print("\n\n✅ LLM Client 双模式测试完美通过！")
        
    except Exception as e:
        print(f"❌ 测试发生异常: {e}")