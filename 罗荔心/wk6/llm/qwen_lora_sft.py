# llm/qwen_lora_sft.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def format_instruction(example):
    """格式化 SFT 数据：Instruction + Input + Output"""
    return {
        "instruction": "你是命名实体识别专家。从文本中提取所有实体，输出JSON格式。",
        "input": example["text"],
        "output": json.dumps(example["label"], ensure_ascii=False)
    }

def prepare_model_and_data():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,               # 低秩矩阵的秩
        lora_alpha=32,    # 缩放因子
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    # 加载 CLUENER2020 并格式化为指令微调格式
    dataset = load_cluener_for_sft()  # 实现略
    return model, tokenizer, dataset

# 使用 HuggingFace Trainer 进行微调
training_args = TrainingArguments(
    output_dir="./qwen_lora_ner",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    logging_steps=50,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
trainer.train()