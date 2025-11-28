# 在代码开头设置环境变量
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 显存紧的话可以换成更小模型
TRAIN_FILE = "data/train_mock.jsonl"
EVAL_FILE = "data/eval_mock.jsonl"
OUTPUT_DIR = "./qwen2-mock-lora"

def load_tokenizer_and_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,                 # 为了跑得快一点，可以比正式项目小一些
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对 Qwen 类模型常见设置
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model

def formatting(example, tokenizer, max_source_len=512, max_target_len=512):
    ins = example["instruction"]
    inp = example.get("input", "")
    out = example["output"]

    if inp:
        prompt = f"指令：{ins}\n输入：{inp}\n回答："
    else:
        prompt = f"指令：{ins}\n回答："

    # 编码 prompt
    source = tokenizer(
        prompt,
        truncation=True,
        max_length=max_source_len,
    )
    # 编码答案
    target = tokenizer(
        out,
        truncation=True,
        max_length=max_target_len,
        add_special_tokens=False,
    )

    input_ids = source["input_ids"] + target["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)

    # 只让模型在答案部分计算 loss，prompt 部分 label 填 -100
    labels = [-100] * len(source["input_ids"]) + target["input_ids"] + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def main():
    tokenizer, model = load_tokenizer_and_model()

    # 加载数据集
    train_ds = load_dataset("json", data_files=TRAIN_FILE)["train"]
    eval_ds = load_dataset("json", data_files=EVAL_FILE)["train"]

    # 映射成 tokenized 格式
    def _tokenize_fn(examples):
        return formatting(examples, tokenizer)

    train_tokenized = train_ds.map(
        _tokenize_fn,
        remove_columns=train_ds.column_names,
    )
    eval_tokenized = eval_ds.map(
        _tokenize_fn,
        remove_columns=eval_ds.column_names,
    )

    # 训练参数：只跑 1~2 epoch，快速验证
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # 累积 8 个 step 等价于 batch_size=8
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    trainer.train()

    # 只保存 LoRA adapter（体积比较小）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
