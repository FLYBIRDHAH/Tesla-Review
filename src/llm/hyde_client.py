# -*- coding: utf-8 -*-
import os
import sys
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from openai import APIConnectionError, APITimeoutError

# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 从你的全局配置导入常量
from config.settings import LLM_MODEL_NAME, LLM_RETRY_ATTEMPTS, LLM_RETRY_WAIT_SECONDS
# 导入你写好的单例连接池
from src.llm.shared_client import get_shared_client
from config.prompts import Prompts

logger = logging.getLogger(__name__)

# HyDE 专属 Prompt
HYDE_PROMPT = """你是一位Tesla汽车专家，现在请你结合Model 3车辆和新能源电动汽车相关知识回答下列问题.
请给出用户问题的使用方法，详细分析问题原因，返回有用的内容。
{query}
最终的回答请尽可能的精简, 不超过100字:
"""

class LLMHydeClient:
    def __init__(self, model_name=None):
        self.model_name = model_name or LLM_MODEL_NAME
        
        # 直接调用你的 shared_client。不传参则默认使用 settings 里的 LLM_API_URL
        self.client = get_shared_client()
        logger.info(f"🔮 HyDE Client 初始化完成，使用模型: {self.model_name}")

    @retry(
        stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
        wait=wait_fixed(LLM_RETRY_WAIT_SECONDS),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    def generate(self, query: str) -> str:
        """生成假设性文档以增强检索"""
        prompt = Prompts.HYDE.format(query=query)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个有用的人工智能助手."},
                    {"role": "user", "content": prompt}
                ],
                top_p=0.1,         # 低随机性
                temperature=0.01,  # 极低温度稳定输出
                max_tokens=150   # 👈 核心必杀技：强行拔电源！最多只允许它生成 150 个 Token！
            )
            hyde_doc = completion.choices[0].message.content.strip()
            logger.info(f"💡 HyDE 成功生成假设文档 (长度: {len(hyde_doc)})")
            return hyde_doc
        except Exception as e:
            logger.error(f"❌ HyDE 生成失败: {e}")
            return ""

# ================= 单元测试 =================
if __name__ == "__main__":
    # 配置基础日志格式，方便看测试输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- 🚀 启动 HyDE Client 测试 ---")
    
    try:
        hyde = LLMHydeClient()
        test_query = "如何调节后视镜？"
        print(f"\n🙋 原始用户提问: {test_query}")
        
        # 调用大模型生成假设文档
        fake_doc = hyde.generate(test_query)
        
        print(f"\n✨ 生成的假设性增强文档:\n{fake_doc}\n")
        print("✅ HyDE 测试完美通过！")
    except Exception as e:
        print(f"❌ 测试发生异常: {e}")