# -*- coding: utf-8 -*-
import os
import sys
import logging
import concurrent.futures
from typing import List, Optional
from tqdm import tqdm
import re  # <--- 新增这行，引入正则表达式模块
from langchain_core.documents import Document

# 确保能跨目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import CLEAN_MAX_WORKERS, LLM_MODEL_NAME
from config.prompts import Prompts
# 导入咱们刚写好的高级单例客户端！
from src.llm.shared_client import get_shared_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMCleanClient:
    def __init__(self, max_workers: int = None):
        """初始化智能清洗客户端"""
        self.model_name = LLM_MODEL_NAME
        self.max_workers = max_workers or CLEAN_MAX_WORKERS
        # 魔法：直接获取全局唯一的复用客户端！告别重复创建连接！
        self.client = get_shared_client()

    def _chat(self, doc_content: str) -> Optional[str]:
        """ 调用大模型进行单段文本的清洗总结 """
        try:
            # 魔法：调用外部统一配置的 Prompt 模板！
            prompt_text = Prompts.CLEAN.format(doc_content=doc_content)
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                top_p=0.1,         # 严谨模式
                temperature=0.01 
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            raise e

    def clean_docs(self, docs: List[Document]) -> List[Document]:
        """ 多线程并发处理文档清洗任务 """
        clean_docs = []
        print(f"🚀 启动智能清洗引擎，并发线程数: {self.max_workers}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._chat, doc.page_content): doc for doc in docs}
            
            # 炫酷进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_doc), total=len(docs), desc="🧠 AI 洗稿中"):
                doc = future_to_doc[future]
                try:
                    res = future.result()
                    if res:
                        clean_docs.append(Document(page_content=res, metadata=doc.metadata))
                except Exception as e:
                    logger.error(f"清洗某页文档时发生致命错误: {e}")

        print("✨ 智能清洗全部完成！")
        return clean_docs

# ================= 测试流水线 =================
if __name__ == "__main__":
    try:
        print("--- 🔧 开始 LLMCleanClient 独立模块测试 ---")
        
        # 1. 构造极度肮脏的测试数据
        dirty_docs = [
            Document(
                page_content="第12页\n\n特\n斯拉 Model  3 的\n\n电池在...底盘下方。\n用户手册\n", 
                metadata={"page": 12}
            )
            #Document(
                #page_content="空调滤芯   需要   每年 更换。\n\nTesla Model 3 用户手册", 
                #metadata={"page": 15}
            #)
        ]
        
        print(f"📥 收到 {len(dirty_docs)} 页待清洗文档...")
        
        # 2. 实例化并运行清洗
        cleaner = LLMCleanClient()
        result_docs = cleaner.clean_docs(dirty_docs)
        
        # 3. 打印对比结果
        print("\n--- 🧼 清洗结果对比 ---")
        for doc in result_docs:
            print(f"📄 页码: {doc.metadata.get('page', '未知')}")
            print(f"✅ 干净内容: \n{doc.page_content}\n" + "-"*30)
            
    except Exception as e:
        print(f"❌ 测试失败，请检查 vLLM 服务是否启动或网络配置: {e}.")