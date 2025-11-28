# 基于 LoRA 的微调与检索增强（RAG）系统

本文档概述本仓库中实现的基于 LoRA（Low-Rank Adaptation）微调与检索增强生成（RAG）示例工程，包含训练、构建 FAISS 向量索引、以及提供一个简单的本地推理/演示服务的说明。

**仓库目标**：提供一个端到端的小型示例，演示如何用 LoRA 对大模型进行轻量微调，并结合基于向量检索的知识库实现 RAG 工作流。

**适用场景**：学习、实验、演示和小规模原型验证。

---

**目录结构（重点）**
- **`app.py`**: 本地演示/服务脚本，启动推理接口或小型服务（详见脚本注释）。
- **`train_lora_mock.py`**: 用于对模型进行 LoRA 微调的训练脚本（演示用的 mock 示例）。
- **`rag_model.py`**, **`rag_model_new.py`**: RAG 管道实现文件（可能包含旧版/新版实现，具体以代码注释为准）。
- **`build_faiss_index.py`**: 将文本/文档构建为 FAISS 向量索引的脚本，输出为 `faiss_index/index.faiss`。
- **`gen_mock_data.py`**: 生成演示用的 mock 数据（写入 `data/`）。
- **`data/`**: 演示数据目录（包含 `train_mock.jsonl`、`eval_mock.jsonl` 等）。
- **`docs/`**: 用作知识库的文本文件（如 `kb_faq.txt`、`kb_product_intro.txt`）。
- **`faiss_index/`**: 存放已经生成的索引文件（如 `index.faiss`）。
- **`qwen2-mock-lora/`**: 本地模型/LoRA 权重与 tokenizer 相关文件（示例权重 `adapter_model.safetensors`、`checkpoint-100/` 等）。
- **`index.html`**: 简单前端页面，用于本地演示（如果 `app.py` 提供前端支持，可打开此页面）。
- **`requirements.txt`**: Python 依赖列表。

---

**快速开始（Windows PowerShell）**

- **环境准备**：建议使用 Python 3.8+（根据 `requirements.txt` 调整）。创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **生成演示数据（可选）**：

```powershell
python gen_mock_data.py
```

- **训练（LoRA 微调）示例**：

```powershell
python train_lora_mock.py
```

训练脚本是演示用途，具体参数（如 batch size、学习率、模型路径、输出目录等）请在脚本开头或内部注释中查看并按需修改。

- **构建 FAISS 索引**：

```powershell
python build_faiss_index.py
```

脚本会读取 `docs/` 或 `data/` 中的源文本（具体逻辑见脚本），并在 `faiss_index/` 下输出 `index.faiss` 文件供检索使用。

- **启动本地演示/服务**：

```powershell
python app.py
```

运行后，`app.py` 会在控制台输出访问地址（例如 `http://localhost:5000` 或其它端口），前端页面 `index.html` 可直接打开或由服务托管。

---

**关键文件说明**
- **`train_lora_mock.py`**：演示如何用 LoRA 方式微调下游任务，包含训练循环、保存 adapter 权重与检查点。
- **`qwen2-mock-lora/`**：示例模型目录，包含 tokenizer、adapter 权重（`adapter_model.safetensors`）和若干 checkpoint，用作载入/微调参考。
- **`build_faiss_index.py`**：负责文本切分、编码（embedding）与 FAISS 索引构建。若要替换向量化模型或参数，请在脚本中对应位置修改。
- **`rag_model.py` / `rag_model_new.py`**：实现检索-生成（RAG）逻辑，将检索到的知识片段与输入拼接后送入生成模型，注意两者可能为不同的实现版本。

---

**使用提示与调优建议**
- 如果使用 GPU，请确保 CUDA 环境与对应的 PyTorch 版本匹配，并在运行脚本前激活 GPU 环境。
- 大模型微调时请注意显存与 batch size 的权衡；LoRA 可以明显降低训练时的显存占用，但仍受主模型与 tokenizer 的影响。
- 构建 FAISS 索引前，请先规范化/清洗文本并选择合适的 embedding 模型（embedding 的维度必须与索引设置一致）。

---

**常见问题（FAQ）**
- 我没有 GPU，能否跑？: 可以运行 CPU 版本做功能验证，但速度会慢且训练/推理受限。
- 如何替换模型？: 将模型路径与 tokenizer 配置替换为目标模型，并在训练/推理脚本中更新加载代码。
- 索引不生效/检索结果差：检查 embedding 模型是否正确、文本是否被合理切分、以及索引和查询是否使用了相同的向量化方法。

---

**贡献**
欢迎提出问题、改进 PR 或补充更完善的训练/评估脚本。请在提交前运行并自测相关变更，尽量保持仓库风格一致。

---

**许可与免责声明**
本仓库为学习/演示用途示例。使用时请确保遵守所用模型与数据的许可协议与使用条款。对于任何因使用本代码产生的问题或损失，作者不承担法律责任。

---

如果你希望我把 `README.md` 中的某个部分扩展为更详细的操作指南（例如：把 `train_lora_mock.py` 的命令行参数与示例命令写清楚，或给出 `app.py` 的端到端演示流程），告诉我你想先完善哪一部分，我会继续更新。
