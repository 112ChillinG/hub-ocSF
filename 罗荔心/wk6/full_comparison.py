#!/usr/bin/env python
# full_comparison.py - 一键对比四种文本分类/NER方法（CLUENER2020）
# 修复了：数据集路径自动定位、batch padding、AdamW导入、HF镜像

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 使用国内镜像

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm
import requests
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torchcrf import CRF
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
# 自动获取脚本所在目录，保证数据集路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "cluener_public")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
DEV_JSON = os.path.join(DATA_DIR, "dev.json")
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS_BERT = 3
LR_BERT = 2e-5

# Qwen API 配置（需要设置环境变量或直接修改）
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen-plus"

# LoRA 配置
LORA_OUTPUT_DIR = os.path.join(script_dir, "qwen_lora_ner")
LORA_EPOCHS = 1
LORA_BATCH_SIZE = 8
SKIP_LORA = True   # 若无GPU或不想跑LoRA，设为True

# 实体类型
ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

# 标签映射（BIO）
tag2id = {"O": 0}
for i, et in enumerate(ENTITY_TYPES):
    tag2id[f"B-{et}"] = 2*i + 1
    tag2id[f"I-{et}"] = 2*i + 2
id2tag = {v: k for k, v in tag2id.items()}
NUM_LABELS = len(tag2id)

# ==================== 1. 数据加载 ====================
def load_raw_data(json_path):
    """读取原始CLUENER文件（每行一个JSON）"""
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_bio(text, entities):
    """将span标注转为BIO序列（按字符）"""
    tokens = list(text)
    labels = ["O"] * len(tokens)
    for etype, spans_dict in entities.items():
        for span_list in spans_dict.values():
            for start, end in span_list:
                if labels[start] == "O":
                    labels[start] = f"B-{etype}"
                for i in range(start+1, end+1):
                    labels[i] = f"I-{etype}"
    return tokens, labels

class CLUENERDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=MAX_LEN):
        self.data = []
        raw = load_raw_data(json_path)
        for item in raw:
            text = item["text"]
            tokens, bio_labels = convert_to_bio(text, item["label"])
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            label_ids = [tag2id[lbl] for lbl in bio_labels]
            # 截断
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                label_ids = label_ids[:max_len]
            self.data.append({
                "input_ids": input_ids,
                "labels": label_ids
            })
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """自定义collate函数：对batch内的序列进行padding，使长度一致"""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    # 找出batch中最长序列长度
    max_len = max(ids.size(0) for ids in input_ids)
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    for ids, lbls in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        padded_lbls = torch.cat([lbls, torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([torch.ones(ids.size(0)), torch.zeros(pad_len)])
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbls)
        attention_masks.append(mask)
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_labels)
    }

# ==================== 2. 模型定义 ====================
class BertLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, NUM_LABELS), labels.view(-1))
        return loss, logits

class BertCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)
        self.crf = CRF(NUM_LABELS, batch_first=True)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            # 注意：CRF需要mask，且要求labels中padding位置为0（但O的id就是0，这里没问题）
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
        return loss, emissions
    def decode(self, input_ids, attention_mask):
        _, emissions = self.forward(input_ids, attention_mask)
        return self.crf.decode(emissions, mask=attention_mask.bool())

# ==================== 3. 训练与评估 ====================
def train_bert_model(model, train_loader, dev_loader, device, epochs=EPOCHS_BERT):
    model.to(device)
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=LR_BERT)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss, _ = model(input_ids, attn_mask, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(train_loader)
        f1 = evaluate_bert_model(model, dev_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Dev F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def evaluate_bert_model(model, dev_loader, device):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if isinstance(model, BertCRF):
                pred_ids = model.decode(input_ids, attn_mask)
                for i, seq in enumerate(pred_ids):
                    seq_len = attn_mask[i].sum().item()
                    pred_tags = [id2tag.get(pid, "O") for pid in seq[:seq_len]]
                    pred_labels.append(pred_tags)
                    true_tags = [id2tag.get(labels[i][j].item(), "O") for j in range(seq_len)]
                    true_labels.append(true_tags)
            else:
                _, logits = model(input_ids, attn_mask)
                pred_ids = logits.argmax(dim=-1)
                for i in range(pred_ids.size(0)):
                    seq_len = int(attn_mask[i].sum().item())
                    pred_tags = [id2tag.get(pred_ids[i][j].item(), "O") for j in range(seq_len)]
                    pred_labels.append(pred_tags)
                    true_tags = [id2tag.get(labels[i][j].item(), "O") for j in range(seq_len)]
                    true_labels.append(true_tags)
    return f1_score(true_labels, pred_labels)

# ==================== 4. Qwen API few-shot ====================
def qwen_api_fewshot(text, examples):
    client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    example_str = ""
    for ex in examples:
        example_str += f"文本：{ex['text']}\n输出：{json.dumps(ex['entities'], ensure_ascii=False)}\n\n"
    prompt = f"""你是命名实体识别专家。遵循以下示例的格式，从输入文本中提取实体。

{example_str}
实体类型：{', '.join(ENTITY_TYPES)}

文本：{text}
输出："""
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    content = response.choices[0].message.content
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except:
        pass
    return {}

def evaluate_qwen_api(dev_data, num_samples=100):
    import random
    random.seed(42)
    samples = dev_data[:min(num_samples, len(dev_data))]
    train_data = load_raw_data(TRAIN_JSON)
    examples = []
    for item in train_data[:3]:
        entities = {}
        for etype, spans_dict in item["label"].items():
            texts = list(spans_dict.keys())
            if texts:
                entities[etype] = [texts[0]]
        examples.append({"text": item["text"], "entities": entities})
    total_true, total_pred, correct = 0, 0, 0
    for item in tqdm(samples, desc="Qwen API评估"):
        text = item["text"]
        true_entities = {}
        for etype, spans_dict in item["label"].items():
            texts = list(set(spans_dict.keys()))
            true_entities[etype] = texts
        pred_entities = qwen_api_fewshot(text, examples)
        for etype in ENTITY_TYPES:
            true_set = set(true_entities.get(etype, []))
            pred_set = set(pred_entities.get(etype, []))
            total_true += len(true_set)
            total_pred += len(pred_set)
            correct += len(true_set & pred_set)
    if total_pred == 0:
        return 0.0
    precision = correct / total_pred
    recall = correct / total_true
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    return f1

# ==================== 5. Qwen LoRA SFT ====================
def train_lora_qwen():
    if SKIP_LORA:
        return None
    train_raw = load_raw_data(TRAIN_JSON)
    def format_instruction(item):
        instruction = "你是命名实体识别专家。从文本中提取所有实体，输出JSON格式。"
        input_text = item["text"]
        entities = {}
        for etype, spans_dict in item["label"].items():
            texts = list(set(spans_dict.keys()))
            if texts:
                entities[etype] = texts
        output = json.dumps(entities, ensure_ascii=False)
        return f"{instruction}\n输入：{input_text}\n输出：{output}"
    train_texts = [format_instruction(item) for item in train_raw]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize(texts, max_len=512):
        return tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.bfloat16, device_map="auto")
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.1)
    model = get_peft_model(model, lora_config)
    from transformers import Trainer, TrainingArguments
    train_dataset = torch.utils.data.TensorDataset(tokenize(train_texts)["input_ids"], tokenize(train_texts)["attention_mask"])
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        learning_rate=2e-4,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    return model

def evaluate_lora_qwen():
    if SKIP_LORA or not os.path.exists(LORA_OUTPUT_DIR):
        return None
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LORA_OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    dev_raw = load_raw_data(DEV_JSON)
    samples = dev_raw[:100]
    correct, total_true, total_pred = 0, 0, 0
    for item in tqdm(samples, desc="LoRA评估"):
        text = item["text"]
        prompt = f"你是命名实体识别专家。从文本中提取所有实体，输出JSON格式。\n输入：{text}\n输出："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            start = pred_text.find('{')
            end = pred_text.rfind('}') + 1
            if start != -1 and end != 0:
                pred_entities = json.loads(pred_text[start:end])
            else:
                pred_entities = {}
        except:
            pred_entities = {}
        true_entities = {}
        for etype, spans_dict in item["label"].items():
            texts = list(set(spans_dict.keys()))
            true_entities[etype] = texts
        for etype in ENTITY_TYPES:
            true_set = set(true_entities.get(etype, []))
            pred_set = set(pred_entities.get(etype, []))
            total_true += len(true_set)
            total_pred += len(pred_set)
            correct += len(true_set & pred_set)
    if total_pred == 0:
        return 0.0
    p = correct / total_pred
    r = correct / total_true
    return 2*p*r/(p+r) if (p+r)>0 else 0

# ==================== 6. 主函数 ====================
def main():
    print("="*60)
    print("CLUENER2020 四种NER方法对比实验")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 检查数据集文件是否存在
    if not os.path.exists(TRAIN_JSON):
        print(f"错误：找不到训练集文件 {TRAIN_JSON}")
        print("请确保数据集放在正确位置：")
        print(f"   {DATA_DIR}/train.json")
        print(f"   {DATA_DIR}/dev.json")
        return
    # 加载数据
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print("加载训练集...")
    train_dataset = CLUENERDataset(TRAIN_JSON, tokenizer)
    dev_dataset = CLUENERDataset(DEV_JSON, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    results = {}
    # 方法1: BERT+Linear
    print("\n[1/4] 训练 BERT+Linear ...")
    model_linear = BertLinear()
    f1_linear = train_bert_model(model_linear, train_loader, dev_loader, device, epochs=EPOCHS_BERT)
    results["BERT+Linear"] = f1_linear
    print(f"✅ BERT+Linear 最终 F1: {f1_linear:.4f}")
    # 方法2: BERT+CRF
    print("\n[2/4] 训练 BERT+CRF ...")
    model_crf = BertCRF()
    f1_crf = train_bert_model(model_crf, train_loader, dev_loader, device, epochs=EPOCHS_BERT)
    results["BERT+CRF"] = f1_crf
    print(f"✅ BERT+CRF 最终 F1: {f1_crf:.4f}")
    # 方法3: Qwen API few-shot
    print("\n[3/4] 评估 Qwen API few-shot ...")
    if QWEN_API_KEY == "your-api-key-here":
        print("⚠️ 未设置有效的 DASHSCOPE_API_KEY，跳过Qwen API评估")
        results["Qwen API (few-shot)"] = None
    else:
        dev_raw = load_raw_data(DEV_JSON)
        f1_api = evaluate_qwen_api(dev_raw, num_samples=100)
        results["Qwen API (few-shot)"] = f1_api
        print(f"✅ Qwen API few-shot F1: {f1_api:.4f}")
    # 方法4: Qwen LoRA SFT
    print("\n[4/4] 处理 Qwen LoRA SFT ...")
    if SKIP_LORA:
        results["Qwen LoRA SFT"] = None
        print("跳过LoRA微调")
    else:
        if not os.path.exists(LORA_OUTPUT_DIR):
            print("开始LoRA微调（可能需要几分钟）...")
            train_lora_qwen()
        print("评估LoRA模型...")
        f1_lora = evaluate_lora_qwen()
        results["Qwen LoRA SFT"] = f1_lora
        print(f"✅ LoRA 微调 F1: {f1_lora:.4f}")
    # 打印对比表格
    print("\n" + "="*60)
    print("最终对比结果（验证集实体级Micro F1）")
    print("="*60)
    print(f"{'方法':<25} {'F1分数':>10}")
    for name, f1 in results.items():
        if f1 is None:
            print(f"{name:<25} {'未运行':>10}")
        else:
            print(f"{name:<25} {f1:>10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()