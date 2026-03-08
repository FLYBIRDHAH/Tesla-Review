# -*- coding: utf-8 -*-
import os
import sys

# 【关键】把项目根目录加入系统路径，这样 Python 才能找到 config 文件夹
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import uvicorn

# 从配置中心引入本地模型路径！
from config.settings import EMBEDDING_M3E_SMALL_DIR

app = FastAPI(title="Tesla-RAG Semantic Chunking API")

print(f"⏳ 正在从配置路径加载模型: {EMBEDDING_M3E_SMALL_DIR}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_M3E_SMALL_DIR)
semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
print("✅ 模型加载完毕，语义分块服务已在 6000 端口就绪！")

class ChunkRequest(BaseModel):
    sentences: str
    group_size: int = 10 

@app.post("/v1/semantic-chunks")
async def chunk_text_api(request: ChunkRequest):
    try:
        text = request.sentences.strip()
        if not text:
            return {"chunks": []}
        
        docs = semantic_chunker.create_documents([text])
        chunks = [doc.page_content for doc in docs]
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)