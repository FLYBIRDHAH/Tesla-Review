# -*- coding: utf-8 -*-
import os
import sys
import uuid
import pickle
import logging
import numpy as np  # 终极数据转换神器
from tqdm import tqdm

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import SPLIT_DOCS_FILE, EMBEDDING_BGE_M3_DIR, EMBEDDING_BATCH_SIZE
from src.database.mongodb_manager import TeslaMongoManager
from src.database.milvus_manager import TeslaMilvusManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_dual_engine_index():
    logger.info("🚀 启动双引擎灌库流水线 (大厂防弹装甲版)！")

    # 1. 拿仓库钥匙 (并清理旧的脏数据)
    mongo_col = TeslaMongoManager.get_collection("manual_text")
    logger.info("🧹 正在清空 MongoDB 旧数据...")
    mongo_col.delete_many({})
    
    milvus_col = TeslaMilvusManager.init_collection(drop_old=True)
    
    if not os.path.exists(SPLIT_DOCS_FILE):
        logger.error(f"❌ 找不到切块文件: {SPLIT_DOCS_FILE}")
        return
        
    with open(SPLIT_DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    logger.info(f"📦 成功加载 {len(docs)} 个文本碎块。")

    # 2. 召唤大模型
    logger.info(f"🧠 正在加载 BGE-M3 旗舰模型...")
    bge_m3 = BGEM3EmbeddingFunction(
        model_name=EMBEDDING_BGE_M3_DIR,
        device="cuda",
        use_fp16=True
    )

    total_batches = (len(docs) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    logger.info("⏳ 正在进行双路向量化与双写灌库...")
    for i in tqdm(range(0, len(docs), EMBEDDING_BATCH_SIZE), total=total_batches, desc="灌库进度"):
        batch_docs = docs[i : i + EMBEDDING_BATCH_SIZE]
        
        batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
        batch_texts = [doc.page_content for doc in batch_docs]
        
        # --- 右手：写 MongoDB (血肉) ---
        mongo_records = [
            {"unique_id": uid, "page_content": doc.page_content, "metadata": doc.metadata}
            for uid, doc in zip(batch_ids, batch_docs)
        ]
        mongo_col.insert_many(mongo_records)
        
        # --- 左手：写 Milvus (灵魂) ---
        embeddings = bge_m3(batch_texts)
        
        # 【防弹转换 1】：密集向量强转 32 位浮点
        dense_float32 = [np.array(v, dtype=np.float32) for v in embeddings["dense"]]
        
        # 【防弹转换 2】：稀疏向量“降维打击”，暴力切成纯 Python 字典！
        sparse_raw = embeddings["sparse"]
        sparse_dicts = []
        
        if hasattr(sparse_raw, "tocsr"):
            csr_mat = sparse_raw.tocsr()
            # 【终极奥义】：不调用任何外部方法，直接扒开 CSR 矩阵的底层内存指针！
            for r in range(csr_mat.shape[0]):
                start_ptr = csr_mat.indptr[r]
                end_ptr = csr_mat.indptr[r+1]
                # 极其关键：顺手把 numpy 的数字强转为 Python 原生的 int 和 float
                # 这样 Milvus 连数据类型报错的机会都没有了！
                row_dict = {
                    int(idx): float(val) 
                    for idx, val in zip(csr_mat.indices[start_ptr:end_ptr], csr_mat.data[start_ptr:end_ptr])
                }
                sparse_dicts.append(row_dict)
                
        elif isinstance(sparse_raw, list) or isinstance(sparse_raw, np.ndarray):
            for item in sparse_raw:
                if hasattr(item, "tocsr"):
                    row_csr = item.tocsr()
                    row_dict = {
                        int(idx): float(val) 
                        for idx, val in zip(row_csr.indices, row_csr.data)
                    }
                    sparse_dicts.append(row_dict)
                elif isinstance(item, dict):
                    sparse_dicts.append({int(k): float(v) for k, v in item.items()})
        
        # 终极入库：将洗得干干净净、整整齐齐的数据交给 Milvus
        milvus_col.insert([
            batch_ids,
            sparse_dicts,   # 绝对安全的字典列表
            dense_float32   # 绝对安全的 float32 列表
        ])

    # 3. 终极落盘
    logger.info("💾 正在执行 Milvus 数据落盘 (Flush)...")
    milvus_col.flush()
    logger.info(f"✅ Milvus 灌库完成，当前向量总数: {milvus_col.num_entities}")
    
    mongo_col.create_index("unique_id", unique=True)
    logger.info("🎉 恭喜！企业级双引擎知识库彻底构建完毕！")

if __name__ == "__main__":
    try:
        build_dual_engine_index()
    except Exception as e:
        logger.error(f"❌ 流水线崩溃了: {e}")
    finally:
        TeslaMongoManager.close()