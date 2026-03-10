# -*- coding: utf-8 -*-
import os
import sys
import pickle
from langchain_core.documents import Document

# 确保能跨目录导入
# 退三步，直达项目根目录 Tesla-Review
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入咱们的三大神器
from src.parser.pdf_loader import TeslaPDFParser
from src.llm.clean_client_llm import LLMCleanClient
from src.llm.chunking_client import ChunkingClient

# 设定存档目录
PROCESSED_DATA_DIR = "./data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def run_pipeline():
    print("🌟 [Tesla-RAG] 全自动数据流水线启动！🌟\n")
    
    # ==========================================
    # 阶段 1：提取 (PDF -> 粗糙文本)
    # ==========================================
    print("--- ✂️ 阶段 1: PDF 智能解析 ---")
    parser = TeslaPDFParser()
    raw_pages = parser.parse()

    # 【格式转换器】把字典转成 Langchain 标准的 Document
    raw_docs = []
    for page_data in raw_pages:
        if page_data.get("text"):
            # 把页码和图片路径藏在 metadata 里，方便以后给大模型溯源
            doc = Document(
                page_content=page_data["text"],
                metadata={"page": page_data["page"], "images": page_data.get("images", [])}
            )
            raw_docs.append(doc)
            
    print(f"✅ 成功包装 {len(raw_docs)} 页待清洗文档。")

    # ==========================================
    # 阶段 2：清洗 (粗糙文本 -> 纯净文本)
    # ==========================================
    print("\n--- 🧠 阶段 2: 大模型智能清洗 ---")
    cleaner = LLMCleanClient()
    cleaned_docs = cleaner.clean_docs(raw_docs)

    # 存个档 (对应你流程图里的 clean_docs.pkl)
    clean_pkl_path = os.path.join(PROCESSED_DATA_DIR, "clean_docs.pkl")
    with open(clean_pkl_path, "wb") as f:
        pickle.dump(cleaned_docs, f)
    print(f"💾 清洗成果已存档至: {clean_pkl_path}")

    # ==========================================
    # 阶段 3：分块 (纯净长文本 -> 语义碎块)
    # ==========================================
    print("\n--- 🔪 阶段 3: 语义分块 ---")
    # 提示：如果你 AutoDL 上的 M3E 服务开着，可以改成 strategy="semantic"
    chunker = ChunkingClient(strategy="semantic") 
    split_docs = []

    for doc in cleaned_docs:
        # 把干净的文本喂给分块器
        chunks = chunker.chunk_text(doc.page_content)
        
        for chunk_text in chunks:
            if chunk_text.strip():
                # 【极其重要】生成新碎片时，必须继承原有的 metadata（比如页码），
                # 这样将来检索出这句话时，我们才知道它来自 PDF 的哪一页！
                split_doc = Document(
                    page_content=chunk_text, 
                    metadata=doc.metadata.copy() 
                )
                split_docs.append(split_doc)

    # 存个档 (对应你流程图里的 split_docs.pkl)
    split_pkl_path = os.path.join(PROCESSED_DATA_DIR, "split_docs.pkl")
    with open(split_pkl_path, "wb") as f:
        pickle.dump(split_docs, f)
        
    print(f"💾 分块完成！共切出 {len(split_docs)} 个完美数据块，已存档至: {split_pkl_path}")
    print("\n🎉 流水线执行完毕！数据已准备好注入向量数据库！")

if __name__ == "__main__":
    run_pipeline()