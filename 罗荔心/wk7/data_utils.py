import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# ---------- 全局标签映射 ----------
label_names = json.load(open("label_names.json", "r", encoding="utf-8"))
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for idx, label in enumerate(label_names)}
num_labels = len(label_names)

# ---------- 数据集类 ----------
class PeopleNERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # 读取 JSON 文件（最外层是一个列表）
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        for sample in raw_data:
            # 原始 tokens 是字列表，拼接成字符串
            text = "".join(sample["tokens"])
            # 将标签字符串转为对应的 ID
            labels = [label2id[tag] for tag in sample["ner_tags"]]
            self.data.append((text, labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, labels = self.data[idx]
        
        # 1. 调用 Tokenizer，开启 offset_mapping
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,                # 不在内部填充，由 DataLoader 统一处理
            max_length=self.max_len,
            return_offsets_mapping=True,  # 核心：获取每个 token 在原文中的位置
            return_tensors=None           # 返回 Python 列表，方便我们手动处理
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]
        
        # 2. 【关键步骤】将 token IDs 转为字符串，用于检测子词（如 "##ple"）
        #    这样比 tokenizer.decode 更准确，且不会产生多余字符
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # 3. 标签对齐逻辑（严谨版）
        aligned_labels = []
        label_idx = 0  # 这个指针指向原始 labels 列表
        
        for i, (token, offset) in enumerate(zip(tokens, offsets)):
            start, end = offset
            
            # 情况 A：特殊符号（[CLS]、[SEP] 或 padding 占位符）
            # 它们的偏移量都是 (0, 0)
            if start == 0 and end == 0:
                aligned_labels.append(-100)  # 忽略该位置的损失计算
                continue
            
            # 情况 B：子词（Subword），例如 "##ple"
            # 注意：中文 BERT 极少拆分汉字，但为了兼容英文/数字，必须保留此逻辑
            if token.startswith("##"):
                # 重复上一个有效标签，且不移动 label_idx 指针
                # 但需要防御：如果 aligned_labels 为空（极罕见），则补 -100
                if aligned_labels:
                    aligned_labels.append(aligned_labels[-1])
                else:
                    aligned_labels.append(-100)
                continue
            
            # 情况 C：普通词（完整词或单个汉字）
            # 取原始标签，并将指针向后移动一位
            if label_idx < len(labels):
                aligned_labels.append(labels[label_idx])
                label_idx += 1
            else:
                # 防御措施：如果因为截断导致标签不够用，补 -100
                aligned_labels.append(-100)
        
        # 4. 长度最终校验（防止意外）
        # 如果 aligned_labels 比 input_ids 长（极少发生），截断到 max_len
        if len(aligned_labels) > self.max_len:
            aligned_labels = aligned_labels[:self.max_len]
        
        # 注意：由于我们设置了 padding=False，实际返回的序列长度可能小于 max_len。
        # DataLoader 中的 collate_fn 会自动填充 input_ids 和 attention_mask，
        # 但 labels 需要我们自己补充 -100 来对齐吗？
        # —— 不需要！因为后续 DataLoader 默认会堆叠成 list，我们会在 train.py 中自定义 collate_fn 来处理。
        # 为了让你直接能用，我把自定义 collate_fn 也写在下面。
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


# ---------- 自定义 DataLoader 的 collate_fn（必备） ----------
# 因为每个样本长度不同，必须用 pad 把它们补齐到相同长度
def collate_fn(batch):
    """
    batch: 列表，每个元素是 __getitem__ 返回的字典
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # 使用 torch.nn.utils.rnn.pad_sequence 进行填充
    # padding_value: input_ids 和 attention_mask 用 0 填充，labels 用 -100 填充
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "labels": labels_padded
    }