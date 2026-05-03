#以文本为输入的多分类任务：
# 对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# 设置随机种子，保证可重复性
def set_seed(seed=64):
    random.seed(seed)#确定文字在文本中的位置
    torch.manual_seed(seed)#RNN/LSTM等权重和bias随机初始化
set_seed()

# ==================== 1. 定义字符集 ====================
# 包含“你”和另外19个常见汉字，共20个字符
CHARS = ['你', '我', '他', '她', '它', '是', '的', '了', '在', '和', 
         '有', '这', '那', '不', '也', '都', '一', '个', '上', '来']
VOCAB_SIZE = len(CHARS)
char_to_idx = {ch: i for i, ch in enumerate(CHARS)}#神经网络只能处理数值，所以需要将字符映射为数字索引
# 其他字符（不含“你”）
OTHER_CHARS = [ch for ch in CHARS if ch != '你']#保证“你”只在文本中出现一次，标签唯一

# ==================== 2. 生成数据集 ====================
def generate_sample():
#生成一条样本：长度为5的字符串，保证有一个“你”，并返回整数序列和标签（0~4）
    pos = random.randint(0, 4)# “你”的位置（0-index）
    #约束生成，而非生成随机句子后再过滤包含0或多个“你”的句子
    seq = []
    for i in range(5):
        if i == pos:
            seq.append(char_to_idx['你'])
        else:
            seq.append(char_to_idx[random.choice(OTHER_CHARS)])
    label = pos# 标签即为位置（0~4）
    return seq, label

def generate_dataset(num_samples):#每次生成一个5字序列+标签
#生成指定数量的样本
    data = []
    for _ in range(num_samples):
        seq, label = generate_sample()
        data.append((torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)))
        #数据预处理：torch.tensor()将列表转换为张量，dtype指定数据类型，long表示整数
    return data

# 生成训练集（8000）和测试集（2000）
train_data = generate_dataset(8000)#样本数及生成句子数
test_data  = generate_dataset(2000)

# 自定义Dataset
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = TextDataset(train_data)
test_dataset  = TextDataset(test_data)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 3. 定义模型（RNN / LSTM）====================
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, model_type='rnn'):
        super().__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be 'rnn' or 'lstm'")
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):#取最后一个时间的输出
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        if self.model_type == 'rnn':
            out, _ = self.rnn(embedded)       # out: (batch, seq_len, hidden_dim)
        else:  # lstm
            out, (h_n, c_n) = self.rnn(embedded)  # out: (batch, seq_len, hidden_dim)
        # 取最后一个时间步的输出
        last_out = out[:, -1, :]              # (batch, hidden_dim)
        logits = self.fc(last_out)            # (batch, num_classes)
        return logits

# ==================== 4. 训练与评估函数 ====================
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in train_loader:#数据加载，预处理中标注的张量打包成batch传入模型
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)#用 labels 计算交叉熵损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
    return acc

# ==================== 5. 分别用RNN和LSTM训练 ====================
# 超参数
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5
EPOCHS = 10

print("="*50)
print("训练 RNN 模型")
rnn_model = SequenceClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, model_type='rnn')
rnn_acc = train_model(rnn_model, train_loader, test_loader, epochs=EPOCHS)

print("\n" + "="*50)
print("训练 LSTM 模型")
lstm_model = SequenceClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, model_type='lstm')
lstm_acc = train_model(lstm_model, train_loader, test_loader, epochs=EPOCHS)

print("\n最终测试准确率：")
print(f"RNN : {rnn_acc:.2f}%")
print(f"LSTM: {lstm_acc:.2f}%")
