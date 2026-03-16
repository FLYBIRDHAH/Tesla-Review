# -*- coding: utf-8 -*-
import sys
import os
import numpy as np  # <--- 新增这行，数据转换神器
import logging
from typing import List
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_core.documents import Document

# 导入全局配置和我们的两把“仓库钥匙”
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import EMBEDDING_BGE_M3_DIR
from src.database.mongodb_manager import TeslaMongoManager
from src.database.milvus_manager import TeslaMilvusManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeslaMilvusRetriever:
    """
    企业级双擎混合检索器 (检索专用，不负责写入)
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        
        # 1. 拿钥匙：连接 MongoDB 与 Milvus
        logger.info("🔌 连接 MongoDB 与 Milvus 数据库...")
        self.mongo_col = TeslaMongoManager.get_collection("manual_text")
        
        # ⚠️ 检索模式下，绝对不能传入 drop_old=True，否则好不容易灌的数据就没了！
        self.milvus_col = TeslaMilvusManager.init_collection(drop_old=False)
        self.milvus_col.load()  # 核心：必须把数据 load 到显存/内存才能被搜索
        
        # 2. 请翻译官：加载 BGE-M3
        logger.info(f"🧠 正在加载 BGE-M3 旗舰模型 ({EMBEDDING_BGE_M3_DIR})...")
        self.embedding_fn = BGEM3EmbeddingFunction(
            model_name=EMBEDDING_BGE_M3_DIR,
            device="cuda",
            use_fp16=True  # 保持和灌库时一样的半精度，保证推理飞快
        )

    def hybrid_search(self, query_dense, query_sparse):
        """
        完美复刻你的架构设计：双路混合检索 + RRF 融合
        """
        # 1. 稠密向量请求（找语义关联）
        dense_req = AnnSearchRequest(
            [query_dense], "dense_vector",
            {"metric_type": "IP", "params": {}},
            limit=self.top_k
        )

        # 2. 稀疏向量请求（找精准关键词）
        sparse_req = AnnSearchRequest(
            [query_sparse], "sparse_vector",
            {"metric_type": "IP", "params": {}},
            limit=self.top_k
        )

        # 3. 终极裁判：RRF 打分融合
        rerank = RRFRanker()
        res = self.milvus_col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=self.top_k,
            output_fields=["unique_id"]  # 关键：只拿 ID，节省 I/O 开销！
        )
        return res[0] # 返回命中结果的列表

    def search(self, query: str) -> List[Document]:
        logger.info(f"🔍 收到用户查询: '{query}'")
        
        # 1. 翻译问题：同时算出两种形态的向量
        query_embeddings = self.embedding_fn.encode_queries([query])
        
        # 👇 核心修复补丁：把提问用的稠密向量也强制转换为 32 位！
        query_dense = np.array(query_embeddings["dense"][0], dtype=np.float32)
        
        query_sparse = query_embeddings["sparse"][[0]]
        
        # 2. 在 Milvus 中执行混合检索
        hybrid_results = self.hybrid_search(query_dense, query_sparse)
        
        # 抽出所有排好序的 ID
        doc_ids = [hit.id for hit in hybrid_results]
        if not doc_ids:
            logger.warning("⚠️ Milvus 未能检索到任何相关内容！")
            return []
            
        # 3. MongoDB 提货：拿着 ID 去找原文本和图片
        logger.info(f"🎯 Milvus 锁定 {len(doc_ids)} 个高分 ID，前往 MongoDB 提取图文...")
        cursor = self.mongo_col.find({"unique_id": {"$in": doc_ids}})
        
        # 因为 MongoDB 返回的数据是无序的，我们必须用一个字典把它们按 ID 存好
        doc_map = {doc["unique_id"]: doc for doc in cursor}
        
        # 4. 组装结果：必须严格按照 Milvus 裁判给出的排名顺序进行组装！
        final_docs = []
        for doc_id in doc_ids:
            if doc_id in doc_map:
                mongo_data = doc_map[doc_id]
                # 将血肉包装成 LangChain 标准的 Document 格式，方便后续送给大模型
                final_docs.append(Document(
                    page_content=mongo_data.get("page_content", ""),
                    metadata=mongo_data.get("metadata", {})
                ))
                
        return final_docs


# ================= 验收测试代码 =================
if __name__ == "__main__":
    print("--- 🚀 启动企业级双擎检索验收测试 ---")
    
    try:
        # 只取排名前 3 的最优结果
        retriever = TeslaMilvusRetriever(top_k=3)
        
        # 你最关心的多模态问题来了！
        test_query = "如何关闭车窗？"
        results = retriever.search(test_query)
        
        print(f"\n✅ 检索完毕，关于问题 '{test_query}' 的 Top-{len(results)} 结果如下：\n")
        
        for i, doc in enumerate(results):
            print(f"🏆 [综合排名 {i+1}]")
            print(f"📄 来源页码: 第 {doc.metadata.get('page')} 页")
            print(f"🖼️ 绑定的配图: {doc.metadata.get('images', [])}")
            print(f"📝 原文内容 (前150字): {doc.page_content[:150]}...\n")
            print("-" * 60)
            
    except Exception as e:
        logger.error(f"❌ 检索过程发生错误: {e}")
    finally:
        TeslaMongoManager.close()