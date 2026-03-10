# -*- coding: utf-8 -*-
import pickle
import os

# 设定你 .pkl 文件的存放路径 (请根据实际情况确认路径是否正确)
PROCESSED_DATA_DIR = "./data/processed"
clean_pkl_path = os.path.join(PROCESSED_DATA_DIR, "clean_docs.pkl")
split_pkl_path = os.path.join(PROCESSED_DATA_DIR, "split_docs.pkl")

def inspect_pkl():
    print("🔍 启动 PKL 开罐器...\n")

    # ==========================================
    # 查看 1：清洗后的完整页面 (clean_docs)
    # ==========================================
    if os.path.exists(clean_pkl_path):
        with open(clean_pkl_path, "rb") as f:
            clean_docs = pickle.load(f)
        print("--- 🧼 [clean_docs.pkl] 清洗结果预览 ---")
        print(f"✅ 总计: {len(clean_docs)} 页文档")
        
        # 随便抽查第 12 页（索引为 11）的数据，防止越界我们加个判断
        sample_idx = min(11, len(clean_docs) - 1) 
        sample_doc = clean_docs[sample_idx]
        
        print(f"👇 抽查第 {sample_idx + 1} 个 Document：")
        print(f"【元数据 Metadata】: {sample_doc.metadata}")
        # 只打印前 150 个字符，免得刷屏
        print(f"【正文 Content (前150字)】: \n{sample_doc.page_content[:150]}...\n")
    else:
        print("⚠️ 找不到 clean_docs.pkl 文件")

    # ==========================================
    # 查看 2：切碎后的语义分块 (split_docs)
    # ==========================================
    if os.path.exists(split_pkl_path):
        with open(split_pkl_path, "rb") as f:
            split_docs = pickle.load(f)
        print("--- 🔪 [split_docs.pkl] 分块结果预览 ---")
        print(f"✅ 总计: {len(split_docs)} 个语义块")
        
        # 抽查前 3 个分块看看切得合不合理
        print("👇 抽查前 3 个分块：")
        for i in range(min(3, len(split_docs))):
            doc = split_docs[i]
            print(f"  [块 {i+1}] Metadata: {doc.metadata}")
            print(f"  [块 {i+1}] Content: {doc.page_content[:100]}...\n")
    else:
        print("⚠️ 找不到 split_docs.pkl 文件")

if __name__ == "__main__":
    inspect_pkl()