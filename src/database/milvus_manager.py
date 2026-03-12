# -*- coding: utf-8 -*-
import os
import sys
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

# 👇 核心改动 1：从你新配置好的 settings.py 中导入全局变量！
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import MILVUS_DB_FILE, MILVUS_COLLECTION_NAME, MILVUS_ID_MAX_LENGTH

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeslaMilvusManager:
    """
    咱们专属的 Milvus 向量数据库管理器 (BGE-M3 双引擎终极版)
    """

    @classmethod
    def connect(cls):
        """连接本地 Milvus Lite 数据库"""
        if not connections.has_connection("default"):
            logger.info(f"⚡ 正在连接本地 Milvus 数据库: {MILVUS_DB_FILE}")
            try:
                connections.connect(uri=MILVUS_DB_FILE)
                logger.info("✅ 成功连接到 Milvus 向量数据库！")
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {e}")
                raise e

    @classmethod
    def init_collection(cls, drop_old: bool = False) -> Collection:
        """初始化向量表 (Collection)"""
        cls.connect()

        # 如果表已经存在，并且要求删掉重建
        if utility.has_collection(MILVUS_COLLECTION_NAME) and drop_old:
            logger.warning(f"🗑️ 发现旧表 {MILVUS_COLLECTION_NAME}，正在执行删除...")
            utility.drop_collection(MILVUS_COLLECTION_NAME)

        # 如果表不存在，我们就开始极其严谨的“建表”操作
        if not utility.has_collection(MILVUS_COLLECTION_NAME):
            logger.info(f"🏗️ 正在创建全新双引擎向量表: {MILVUS_COLLECTION_NAME}")
            
            # 👇 核心改动 2：重新定义表结构 (Schema)，必须加入 sparse_vector！
            fields = [
                # 主键 ID (长度从 settings 导入)
                FieldSchema(name="unique_id", dtype=DataType.VARCHAR, is_primary=True, max_length=MILVUS_ID_MAX_LENGTH),
                # 稀疏向量：存放 BGE-M3 算出来的关键词权重
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                # 密集向量：存放 BGE-M3 算出来的语义向量 (维度固定为 1024)
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            schema = CollectionSchema(fields, description="Tesla BGE-M3 Hybrid Search Collection")

            # 创建表
            col = Collection(name=MILVUS_COLLECTION_NAME, schema=schema)

            # 👇 核心改动 3：为两列向量分别创建不同的底层索引树！
            logger.info("⚙️ 正在为 sparse_vector 构建 SPARSE_INVERTED_INDEX 倒排索引...")
            col.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
            
            logger.info("⚙️ 正在为 dense_vector 构建 AUTOINDEX 智能索引...")
            col.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
            
            logger.info("✅ 向量表与双重索引创建完毕！")
            return col
            
        else:
            logger.info(f"🔍 检测到向量表 {MILVUS_COLLECTION_NAME} 已存在，直接加载。")
            return Collection(MILVUS_COLLECTION_NAME)


# ================= 测试代码 =================
if __name__ == "__main__":
    print("--- 🚀 开始测试 TeslaMilvusManager (双引擎版) ---")
    try:
        # 测试：连接并初始化表 (drop_old=True 确保能看到全新的建表过程)
        col = TeslaMilvusManager.init_collection(drop_old=True)
        col.load()
        print(f"📊 当前表中数据行数: {col.num_entities}")
    except Exception as e:
        print(f"发生错误了: {e}")