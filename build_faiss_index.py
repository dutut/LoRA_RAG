# build_faiss_index.py
# 在代码开头添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

"""
一步离线脚本：
1. 从 ./docs 目录读取所有 .txt 文件
2. 分块
3. 用中文 embedding 模型编码
4. 构建 Faiss 索引并保存到 ./faiss_index
"""

DOCS_DIR = Path("./docs")
INDEX_DIR = Path("./faiss_index")

def load_docs():
    texts = []
    for path in DOCS_DIR.rglob("*.txt"):
        print(f"加载文档: {path}")
        content = path.read_text(encoding="utf-8")
        texts.append(content)
    return texts

def main():
    docs = load_docs()
    if not docs:
        raise RuntimeError("docs 目录下没有找到任何 .txt 文件，请先放一点知识库文本进去。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", "。", "！", "？"]
    )

    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d))

    print(f"总文本块数量: {len(chunks)}")

    # 中文向量模型：bge-small-zh 速度比较快
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_texts(chunks, emb)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"索引已保存到: {INDEX_DIR}")

if __name__ == "__main__":
    main()
