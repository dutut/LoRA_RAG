# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from aiocache import cached, SimpleMemoryCache

from rag_model_new import generate_answer  # 直接复用刚才写的推理模块
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LoRA + RAG + Qwen API")
# 新增：CORS 配置（开发环境直接全放开即可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 开发环境直接 *，生产可以改成具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    contexts: List[str]

# 简单缓存：相同 query 在 60 秒内直接返回上次的结果
@cached(ttl=60, cache=SimpleMemoryCache)
async def rag_answer_cached(query: str):
    answer, ctx = generate_answer(query)
    return {"answer": answer, "contexts": ctx}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await rag_answer_cached(req.query)
    return ChatResponse(**result)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
