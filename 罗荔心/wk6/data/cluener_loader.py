# data/cluener_loader.py

import json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# CLUENER2020的10个实体类别
ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

def build_tag2id(entity_types):
    """构建 BIO 标签映射: B-{type}, I-{type}, O"""
    tag2id = {"O": 0}
    for i, et in enumerate(entity_types):
        tag2id[f"B-{et}"] = 2 * i + 1
        tag2id[f"I-{et}"] = 2 * i + 2
    return tag2id

def convert_to_bio(text, entities, tokenizer):
    """将 CLUENER2020 的实体标注转为 BIO 序列"""
    tokens = list(text)   # 按字分词（BERT中文按字切分）
    labels = ["O"] * len(tokens)
    for entity_type, spans in entities.items():
        for entity, positions in spans.items():
            for start, end in positions:
                if labels[start] == "O":
                    labels[start] = f"B-{entity_type}"
                for i in range(start + 1, end + 1):
                    labels[i] = f"I-{entity_type}"
    return tokens, labels

class CLUENERDataset(Dataset):
    def __init__(self, json_path, tokenizer, tag2id, max_len=128):
        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                tokens, bio_labels = convert_to_bio(item["text"], item["label"], tokenizer)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                label_ids = [tag2id[label] for label in bio_labels]
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    label_ids = label_ids[:max_len]
                self.data.append({
                    "input_ids": input_ids,
                    "labels": label_ids,
                    "attention_mask": [1] * len(input_ids)
                })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]