# rag_model.py
import torch
from pathlib import Path
from typing import List, Tuple
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 添加到代码开头
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ==== 路径 & 模型配置 ====
BASE_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 显存不够可以改成更小模型
LORA_PATH = "./qwen2-mock-lora"             # 你刚刚训练的 LoRA 输出目录
INDEX_DIR = "./faiss_index"                 # 前一步构建的索引目录

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# ==== 全局单例（FastAPI 多请求时都共用同一套模型和索引）====
tokenizer = None
llm = None
vectorstore = None
reranker = None

def init_models():
    global tokenizer, llm, vectorstore, reranker

    if tokenizer is not None:
        return  # 已初始化

    print(">>> 初始化 LLM + LoRA ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    if tokenizer_local.pad_token is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model_with_lora = PeftModel.from_pretrained(base_model, LORA_PATH)
    model_with_lora.eval()

    print(">>> 初始化 Embedding + Faiss 索引 ...")
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    if not Path(INDEX_DIR).exists():
        raise RuntimeError(f"向量索引目录 {INDEX_DIR} 不存在，请先运行 build_faiss_index.py")

    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

    print(">>> 初始化 reranker ...")
    rerank_model = CrossEncoder(RERANK_MODEL_NAME)

    tokenizer = tokenizer_local
    llm = model_with_lora
    vectorstore = vs
    reranker = rerank_model

def retrieve_with_rerank(query: str, top_k: int = 5) -> List[str]:
    """
    先用 Faiss 召回，再用 cross-encoder 精排，返回 top_k 个文档文本
    """
    if vectorstore is None:
        init_models()

    # 初筛 20 个
    docs = vectorstore.similarity_search(query, k=20)
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    final_docs = [d.page_content for d, _ in scored[:top_k]]
    return final_docs

def build_prompt(query: str, contexts: List[str]) -> str:
    context_str = "\n\n".join([f"[文档{i+1}]\n{c}" for i, c in enumerate(contexts)])
    prompt = (
        "你是一个严谨的中文问答助手，请严格基于给定文档回答问题，"
        "如果无法从文档中找到答案，请明确说明“根据提供的文档无法确定答案”，不要编造信息。\n\n"
        f"【检索到的文档】:\n{context_str}\n\n"
        f"【用户问题】：{query}\n\n"
        "【回答】："
    )
    return prompt

@torch.no_grad()
def generate_answer(query: str) -> Tuple[str, List[str]]:
    """
    对外暴露的主函数：
    输入：用户问题
    输出：(答案文本, 使用到的文档列表)
    """
    if llm is None or tokenizer is None or vectorstore is None:
        init_models()

    contexts = retrieve_with_rerank(query, top_k=5)
    prompt = build_prompt(query, contexts)

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    output_ids = llm.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
    )

    # 只解码新生成部分
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    return answer, contexts
