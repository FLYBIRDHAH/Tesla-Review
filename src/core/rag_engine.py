# -*- coding: utf-8 -*-
import os
import sys
import re
import logging
from typing import List, Dict, Any

# 导入咱们辛辛苦苦打造的所有“零部件”
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retriever.advanced_bm25 import AdvancedBM25
from src.retriever.milvus_retriever import TeslaMilvusRetriever
from src.reranker.bge_m3_reranker import BGEM3Reranker
from src.llm.llm_client import LLMClient
from src.llm.hyde_client import LLMHydeClient
from src.database.mongodb_manager import TeslaMongoManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Tesla-RAG 终极检索引擎指挥官
    """
    def __init__(self, use_hyde: bool = True, use_vllm: bool = False):
        logger.info("🚀 正在启动 Tesla-RAG 终极引擎...")
        
        self.use_hyde = use_hyde
        
        # 1. 初始化 LLM 和 HyDE 组件
        self.llm_client = LLMClient(use_vllm=use_vllm)
        self.hyde_client = LLMHydeClient() if use_hyde else None
        
        # 2. 初始化双路检索器
        self.bm25 = AdvancedBM25()
        self.milvus = TeslaMilvusRetriever(top_k=10) # 扩大召回池
        
        # 3. 初始化重排序裁判
        self.reranker = BGEM3Reranker(top_k=5)
        
        logger.info("✅ 引擎所有组件点火完毕！")

    def _merge_docs(self, bm25_docs: list, milvus_docs: list) -> list:
        """物理去重：合并两路召回的结果"""
        merged_pool = {}
        for doc in bm25_docs + milvus_docs:
            # 使用原文本内容作为唯一指纹去重
            merged_pool[doc.page_content] = doc
        unique_docs = list(merged_pool.values())
        logger.info(f"🐟 双路共召回 {len(bm25_docs) + len(milvus_docs)} 条，去重后剩余 {len(unique_docs)} 条候选文档。")
        return unique_docs

    def retrieve_and_rank(self, query: str) -> list:
        """Step 1: 核心检索与重排序流水线"""
        search_query = query
        
        # 1. HyDE 扩展 (可选)
        if self.use_hyde and self.hyde_client:
            logger.info("🪄 正在使用 HyDE 魔法扩写查询...")
            hyde_doc = self.hyde_client.generate(query)
            if hyde_doc:
                # 将大模型的“幻觉猜测”拼接到原问题后面
                search_query = f"{query}\n{hyde_doc}"
                
        # 2. 双路检索
        logger.info("🕸️ 正在执行 BM25 与 Milvus 双路并发检索...")
        bm25_docs = self.bm25.search(search_query, k=5)
        milvus_docs = self.milvus.search(search_query)
        
        # 3. 合并去重
        merged_docs = self._merge_docs(bm25_docs, milvus_docs)
        
        # 4. 重排序打分
        ranked_docs = self.reranker.rerank(query, merged_docs)
        return ranked_docs

    def _post_processing(self, raw_response: str, ranked_docs: list) -> Dict[str, Any]:
        """
        Step 4: 后处理 - 从大模型生成的答案中提取引用编号，并挂载真实的图文信息
        """
        # 匹配大模型输出的【1】, 【1, 2】等引用格式
        cited_indices = set()
        matches = re.findall(r'【([\d,\s]+)】', raw_response)
        for match in matches:
            # 把 "1, 2" 拆开变成数字
            for num_str in match.split(','):
                try:
                    idx = int(num_str.strip()) - 1 # 文档索引从0开始，引用从1开始
                    if 0 <= idx < len(ranked_docs):
                        cited_indices.add(idx)
                except ValueError:
                    continue
        
        # 收集被引用的页码和图片
        reference_pages = set()
        related_images = []
        for idx in cited_indices:
            doc_meta = ranked_docs[idx].metadata
            if "page" in doc_meta:
                reference_pages.add(doc_meta["page"])
            if "images" in doc_meta:
                related_images.extend(doc_meta["images"])
                
        return {
            "answer": raw_response,
            "cite_pages": sorted(list(reference_pages)),
            "related_images": list(set(related_images)), # 图片去重
            "raw_docs": ranked_docs
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """中枢控制：处理用户查询的完整生命周期"""
        logger.info(f"========== 🏁 开始处理请求: '{query}' ==========")
        
        # 1. 检索与重排
        ranked_docs = self.retrieve_and_rank(query)
        if not ranked_docs:
            return {"answer": "抱歉，知识库中未能找到相关信息。", "cite_pages": [], "related_images": []}
            
        # 2. 构建上下文 (加上编号，逼迫大模型引用)
        context_parts = []
        for i, doc in enumerate(ranked_docs):
            context_parts.append(f"[{i+1}] {doc.page_content}")
        context = "\n\n".join(context_parts)
        logger.info("📜 上下文构建完毕，正在呼叫大模型生成最终答案...")
        
        # 3. 大模型生成最终答案 (非流式调用)
        raw_response = self.llm_client.chat(query, context, stream=False)
        
        # 4. 后处理：图文关联
        final_result = self._post_processing(raw_response, ranked_docs)
        logger.info("🎉 请求处理完成！")
        return final_result

# ================= 终极大验收 =================
if __name__ == "__main__":
    print("--- 🌟 Tesla-RAG 引擎启动 ---")
    try:
        # 初始化引擎 (关闭 vLLM 使用远程 API 测试，开启 HyDE 魔法)
        engine = RAGEngine(use_hyde=True, use_vllm=False)
        
        test_queries = [
            "如何加热方向盘？",
            "如果车彻底没电了，怎么从里面把车门打开？"
        ]
        
        for q in test_queries:
            print(f"\n\n🗣️ 用户提问: {q}")
            result = engine.process_query(q)
            
            print(f"\n🤖 终极回答:\n{result['answer']}")
            print(f"\n📚 引用页码: {result['cite_pages']}")
            print(f"🖼️ 关联图片: {result['related_images']}")
            print("-" * 60)
            
    except Exception as e:
        logger.error(f"❌ 引擎运行崩溃: {e}")
    finally:
        # 关闭 MongoDB 连接
        TeslaMongoManager.close()