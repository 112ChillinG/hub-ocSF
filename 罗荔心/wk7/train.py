import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # 新版 PyTorch 已内置 AdamW
from transformers import BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
from data_utils import PeopleNERDataset, label_names, num_labels

# 固定参数
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = PeopleNERDataset("train.json", tokenizer, MAX_LEN)
val_dataset = PeopleNERDataset("validation.json", tokenizer, MAX_LEN)
test_dataset = PeopleNERDataset("test.json", tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

# 优化器 & 调度器
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 评估函数（返回F1）
def evaluate(loader, mode="val"):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {mode}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # 转成可读的标签字符串（忽略 -100）
            for i in range(labels.size(0)):
                true_seq = []
                pred_seq = []
                for j in range(labels.size(1)):
                    if labels[i][j].item() != -100:
                        true_seq.append(label_names[labels[i][j].item()])
                        pred_seq.append(label_names[predictions[i][j].item()])
                true_labels.append(true_seq)
                pred_labels.append(pred_seq)
    
    return f1_score(true_labels, pred_labels), classification_report(true_labels, pred_labels)

# ---------- 训练循环 ----------
print("开始训练...")
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - 平均Loss: {avg_loss:.4f}")
    
    # 验证
    val_f1, val_report = evaluate(val_loader, mode="validation")
    print(f"Epoch {epoch+1} - Validation F1: {val_f1:.4f}")
    
    # 保存最佳模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.bin")
        print(">>> 模型已保存 (最佳F1更新)")

# ---------- 测试集最终评估（单独跑推理） ----------
print("\n加载最佳模型，在测试集上进行最终评估...")
model.load_state_dict(torch.load("best_model.bin"))
test_f1, test_report = evaluate(test_loader, mode="test")
print(f"\n========== 测试集最终结果 ==========")
print(f"F1分数: {test_f1:.4f}")
print(test_report)