import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------- 1. 极简数据准备 -------------------
corpus = [
    "天气真好,我应该去公园散步",
    "我想去遛狗",
    "人工智能很有趣"
]
# 构建字符级词表
vocab = {char: idx for idx, char in enumerate(set(''.join(corpus)))}
#{Char构建字典映射字符(Enumerate生成唯一整数ID(set取字符合集去重（join把输入的句子拼接起来）
vocab_size = len(vocab)

def text_to_ids(text):
    return [vocab[ch] for ch in text]

class TextDataset(Dataset):
    def __init__(self, texts):
        self.data = [torch.tensor(text_to_ids(t), dtype=torch.long) for t in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # 形状 (seq_len,)

# ------------------- 2. 修正后的语言模型 -------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_seq_len=20):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        # 使用 TransformerEncoder 并手动传入因果掩码 (is_causal 需要较新版本，这里兼容写法)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)#堆叠
        self.lm_head = nn.Linear(d_model, vocab_size)#将d_model维的特征映射回词汇表大小，用于预测下一个字符的概率
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape   # idx: (B, T)
        # 生成因果掩码 (上三角为 True，表示屏蔽未来位置)
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        # 位置编码
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_embedding(idx) + self.pos_embedding(positions)
        # TransformerEncoder 需要 src_mask 参数
        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        return logits

# ------------------- 3. 极简训练循环 -------------------
def train(model, dataloader, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # batch 已经是 (B, T)，其中 B=1
            logits = model(batch)          # (1, T, vocab_size)
            # 目标: 把输入向右移一位，忽略最后一个位置的预测（因为没有真实目标）
            target = batch[:, 1:]           # (B, T-1)
            logits = logits[:, :-1, :]      # (B, T-1, V)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                target.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()#记录平均loss
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# ------------------- 4. 直接运行 -------------------
dataset = TextDataset(corpus)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
model = TinyLM(vocab_size)
print("开始训练...")
train(model, dataloader)

# ------------------- 5.验证文本生成功能 -------------------
def generate(model, prompt, max_new_tokens=5):
    model.eval()
    input_ids = torch.tensor([text_to_ids(prompt)]).long()
    for _ in range(max_new_tokens):
        with torch.no_grad():#推理阶段不需要训练，所以不更新参数。不保留中间的激活值。
            logits = model(input_ids)          # (1, T, V)
            next_token_logits = logits[0, -1, :]   # 最后一个位置的输出
            next_token = torch.argmax(next_token_logits).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        # 防止超出位置编码表
        if input_ids.shape[1] >= model.max_seq_len:
            break
    # 将 ids 转回字符串
    inv_vocab = {v: k for k, v in vocab.items()}
    generated = ''.join([inv_vocab[id.item()] for id in input_ids[0]])
    return generated

print(generate(model, "公园"))  
