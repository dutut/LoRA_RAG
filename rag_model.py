import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 添加到代码开头
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "Qwen/Qwen2-7B-Instruct"
lora_path = "./qwen2-mock-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()
