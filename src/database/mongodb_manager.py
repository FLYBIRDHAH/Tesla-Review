# -*- coding: utf-8 -*-
import os
import sys
import pymongo
import logging

# 👇 核心改动 1：从全局配置中心导入连接地址和库名
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import MONGO_URI, MONGO_DB_NAME

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeslaMongoManager:
    """
    咱们专属的 MongoDB 数据库连接管理器 (企业级解耦版)
    """
    _client = None  
    _db = None

    @classmethod
    def get_db(cls):
        """获取数据库实例（懒加载模式）"""
        if cls._client is None:
            # 👇 核心改动 2：日志打印动态读取的地址，方便排错
            logger.info(f"⚡ 正在初始化 MongoDB 连接池 (URI: {MONGO_URI})...")
            try:
                # 👇 核心改动 3：彻底告别硬编码！
                cls._client = pymongo.MongoClient(MONGO_URI)
                cls._db = cls._client[MONGO_DB_NAME]
                
                # ping 测试连通性
                cls._client.admin.command('ping')
                logger.info(f"✅ 成功连接到 MongoDB 数据库: [{MONGO_DB_NAME}]")
                
            except Exception as e:
                logger.error(f"❌ 数据库连接彻底失败，请检查 MongoDB 服务是否开启: {e}")
                raise e
                
        return cls._db

    @classmethod
    def get_collection(cls, collection_name: str):
        """提供给外部的快捷接口：获取指定表（集合）"""
        db = cls.get_db()
        return db[collection_name]

    @classmethod
    def close(cls):
        """优雅关闭连接池"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("👋 MongoDB 连接池已安全关闭。")

# ================= 测试代码 =================
if __name__ == "__main__":
    print("--- 🚀 开始测试 TeslaMongoManager (解耦版) ---")
    
    try:
        # 获取集合
        collection = TeslaMongoManager.get_collection("manual_text")
        
        # 清理旧数据并测试插入
        collection.delete_many({})
        print("🧹 已清空旧测试数据...")
        
        test_doc = {
            "unique_id": "doc-uuid-001",
            "page_content": "这是一条测试洗好的特斯拉文本。",
            "metadata": {"page": 12, "images": ["./data/img/page12_1.png"]}
        }
        
        insert_res = collection.insert_one(test_doc)
        print(f"📥 成功插入 1 条数据！分配的 ID: {insert_res.inserted_id}")
        
        found_doc = collection.find_one({"unique_id": "doc-uuid-001"})
        print(f"\n🔍 查验数据库提取的数据:\n{found_doc}")
        
    except Exception as e:
        print(f"发生错误了: {e}")
    finally:
        TeslaMongoManager.close()