# -*- coding: utf-8 -*-
import os
import sys
import logging
from typing import List
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import RERANK_BGE_M3_DIR
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BGEM3Reranker:
    """
    基于 BGE 架构的 Cross-Encoder 重排序器
    作为 RAG 检索的最后一道防线，精准过滤无关文档，压榨出最强 Top-K
    """
    def __init__(self, model_path: str = RERANK_BGE_M3_DIR, top_k: int = 5):
        self.top_k = top_k
        logger.info(f"⚖️ 正在请出终极裁判: BGE Cross-Encoder ({model_path})...")
        
        # 同样开启 FP16 极致提速，AutoDL 上的 4090 跑这个毫无压力
        self.reranker = FlagReranker(model_path, use_fp16=True)
        logger.info("✅ 裁判已就位！")

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        对初筛上来的文档进行残酷的“交叉审判”
        """
        if not docs:
            logger.warning("⚠️ 没有输入任何候选文档，重排序跳过。")
            return []
            
        logger.info(f"🧐 正在对 {len(docs)} 个候选文档进行深度交叉打分...")
        
        # 1. 组装案卷：把问题和每一个候选文档“绑”在一起 [ [问题, 文本1], [问题, 文本2]... ]
        sentence_pairs = [[query, doc.page_content] for doc in docs]
        
        # 2. 裁判打分：Cross-Encoder 的底层魔法就在这一行
        scores = self.reranker.compute_score(sentence_pairs)
        
        # 兼容性容错：如果只传了一个文档进来，它会返回浮点数而不是列表
        if isinstance(scores, float):
            scores = [scores]
            
        # 3. 将分数与原文档捆绑在一起
        doc_with_scores = list(zip(docs, scores))
        
        # 4. 根据分数从高到低进行残酷淘汰排序
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 选出 Top-K，并把分数打在元数据上（方便我们调试看效果）
        final_docs = []
        for rank, (doc, score) in enumerate(doc_with_scores[:self.top_k]):
            # 顺手将排名和得分烙印在 metadata 里
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank + 1
            final_docs.append(doc)
            
        logger.info(f"🎯 审判结束，已为您精选出 Top-{len(final_docs)} 份绝密情报。")
        return final_docs

# ================= 模拟大合并验收测试 =================
if __name__ == "__main__":
    from src.retriever.advanced_bm25 import AdvancedBM25
    from src.retriever.milvus_retriever import TeslaMilvusRetriever
    
    print("--- 🚀 启动多路召回 + 终极重排序大阅兵 ---")
    
    try:
        # 1. 唤醒两位渔夫 (BM25 负责抓死理，Milvus 负责抓语义)
        # 注意：如果你之前没跑通 BM25，可以只测试 Milvus，这里是为了演示终极合体！
        bm25_retriever = AdvancedBM25()
        milvus_retriever = TeslaMilvusRetriever(top_k=5)
        
        # 2. 唤醒裁判
        reranker = BGEM3Reranker(top_k=3)
        
        test_query = "如何关闭车窗？"
        print(f"\n📢 用户提问: {test_query}")
        
        # 3. 两路齐发，下网捞鱼
        bm25_docs = bm25_retriever.search(test_query, k=5)
        milvus_docs = milvus_retriever.search(test_query)
        
        # 4. 去重合并 (MergeDocs)
        # 我们用原文本做 Key 进行物理去重，防止两个检索器找出了同一段话
        merged_pool = {}
        for doc in bm25_docs + milvus_docs:
            merged_pool[doc.page_content] = doc
        unique_docs = list(merged_pool.values())
        print(f"🐟 两网共捞出 {len(bm25_docs) + len(milvus_docs)} 条记录，去重后剩余 {len(unique_docs)} 条候选！")
        
        # 5. 终极审判
        final_results = reranker.rerank(test_query, unique_docs)
        
        print("\n🏆 最终递交给大模型的金牌资料：\n")
        for doc in final_results:
            print(f"🥇 [最终排名 {doc.metadata['rerank_rank']}] | 评分: {doc.metadata['rerank_score']:.4f}")
            print(f"📄 页码: {doc.metadata.get('page')} | 🖼️ 图片: {doc.metadata.get('images', [])}")
            print(f"📝 内容: {doc.page_content[:100]}...\n")
            
    except Exception as e:
        print(f"❌ 测试发生错误: {e}")