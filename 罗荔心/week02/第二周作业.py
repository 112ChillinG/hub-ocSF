#完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类
#单层线性网络+CrossEntropyLoss（包含softmax+交叉熵）+Adam优化器
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.utils.data as Data

# ========== 1. 生成数据 ==========
input_dim = 10      # 随机向量的维度，也是类别数
num_classes = 10   #类别共10种，与维度对应
num_samples = 50000 # 训练样本数

def generate_data(num_samples, dim):
    X = np.random.randn(num_samples, dim).astype(np.float32)#生成标准正态分布的随机数
    y = np.argmax(X, axis=1).astype(np.int64)#找出每行最大值的索引，作为类别标签
    return torch.from_numpy(X), torch.from_numpy(y)

X_train, y_train = generate_data(num_samples, input_dim)
X_test, y_test = generate_data(10000, input_dim)

# ========== 2. 定义模型 ==========
#单层线性网络-仅全连接层
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

model = LinearClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()#包含softmax+交叉熵，先得到概率分布，再计算与真实标签的差距
optimizer = optim.Adam(model.parameters(), lr=0.001)#Adam优化器，收敛更快

# ========== 3. 训练并记录指标 ==========
epochs = 20
batch_size = 64
dataset = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)#dataloader，将数据集分批加载；同时打乱顺序避免模型存在记忆

train_losses = []
test_accs = []

#每个epoch只做一次梯度更新，所以需要手动清零梯度
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(X_train)
    train_losses.append(avg_loss)
    
    # 测试
    #model.eval()关闭梯度计算，节省内存
    with torch.no_grad():
        logits = model(X_test)
        pred = torch.argmax(logits, dim=1)#前向传播
        acc = (pred == y_test).float().mean().item()#计算准确率
        test_accs.append(acc)
    print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")

# ========== 4. 绘图：损失曲线和准确率曲线 ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), test_accs, marker='s', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Curve')
plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()


# ========== 5. 随机样本预测对比（条形图） ==========
num_show = 20
indices = np.random.choice(len(X_test), num_show, replace=False)
sample_x = X_test[indices]
sample_y_true = y_test[indices]

with torch.no_grad():
    sample_logits = model(sample_x)
    sample_y_pred = torch.argmax(sample_logits, dim=1)

plt.figure(figsize=(12, 4))
x_axis = np.arange(num_show)
width = 0.35
plt.bar(x_axis - width/2, sample_y_true.numpy(), width, label='True Label')
plt.bar(x_axis + width/2, sample_y_pred.numpy(), width, label='Predicted Label')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Comparison of True vs Predicted Classes (Random Test Samples)')
plt.xticks(x_axis)
plt.legend()
plt.grid(axis='y')
plt.show()
