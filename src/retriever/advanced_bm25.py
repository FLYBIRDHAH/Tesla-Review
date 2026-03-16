import pickle
import jieba
import os
from langchain_community.retrievers import BM25Retriever

class AdvancedBM25:
    def __init__(self, pkl_path="./data/processed/split_docs.pkl", index_path="./data/saved_index/bm25.pkl"):
        self.index_path = index_path
        
        # 如果硬盘上有缓存，直接秒加载
        if os.path.exists(self.index_path):
            print("⚡ 检测到本地缓存，直接加载高级 BM25 索引...")
            with open(self.index_path, 'rb') as f:
                self.retriever = pickle.load(f)
            # 恢复自定义的分词函数
            self.retriever.preprocess_func = self.jieba_tokenize
        else:
            # 没有缓存，老老实实加载数据建索引
            print("⚙️ 首次运行，正在使用 Jieba 构建高级 BM25 索引...")
            with open(pkl_path, "rb") as f:
                docs = pickle.load(f)
            self.retriever = BM25Retriever.from_documents(docs, preprocess_func=self.jieba_tokenize)
            
            # 存到硬盘，下次免检
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.retriever, f)
            print("💾 索引已保存到硬盘！")

    def jieba_tokenize(self, text: str):
        # 接入 Jieba 词库，这里可以后续自己加停用词过滤
        return [t for t in jieba.lcut(text) if len(t.strip()) > 0]

    def search(self, query: str, k=5):
        self.retriever.k = k
        return self.retriever.invoke(query)
    # ================= 测试代码 =================
if __name__ == "__main__":
    print("--- 🔧 开始高级 BM25 模块测试 ---")
    
    # 初始化（它会自动找 pkl 文件或者重新建索引）
    try:
        bm25 = AdvancedBM25()
        
        # 模拟用户提问
        test_query = "方向盘怎么加热？"
        print(f"\n🔍 正在检索问题: '{test_query}'")
        
        # 执行检索
        results = bm25.search(test_query, k=2)
        
        for i, doc in enumerate(results):
            print(f"\n[匹配结果 {i+1}]")
            print(f"页码: {doc.metadata.get('page', '未知')}")
            print(f"内容: {doc.page_content[:150]}...")
            
    except FileNotFoundError:
        print("❌ 找不到 split_docs.pkl 文件，请先运行你的 data_pipeline.py 噢！")