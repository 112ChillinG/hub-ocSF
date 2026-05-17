import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Config:
#模型配置，统一管理超参数
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, max_seq_len=5000):
        self.d_model = d_model      # 模型维度，即词嵌入的维度
        self.n_heads = n_heads      # 注意力头数量
        self.d_ff = d_ff            # 前馈网络隐藏层维度
        self.dropout = dropout      # Dropout比率
        self.max_seq_len = max_seq_len # 最大序列长度
        # 确保 d_model 可以被 n_heads 整除
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # 每个注意力头的维度

#自注意力
class ScaledDotProductAttention(nn.Module):
 #缩放点积注意力
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数:
        Q: [batch_size, n_heads, seq_len_q, d_k]
        K: [batch_size, n_heads, seq_len_k, d_k]
        V: [batch_size, n_heads, seq_len_v, d_k]
        mask: [batch_size, 1, seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k], 用于屏蔽特定位置

        返回:
        output: [batch_size, n_heads, seq_len_q, d_k]
        attn: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        # 1. 计算原始注意力分数: Q 和 K 的转置做矩阵乘法, 再除以 sqrt(d_k) 进行缩放
        # 公式: scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

        # 2. 应用掩码 (可选): 将需要屏蔽的位置设置为一个非常大的负数(-1e9),
        # 这样经过 Softmax 后, 这些位置的注意力权重就会接近于 0。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. 应用 Softmax 得到注意力权重
        attn = F.softmax(scores, dim=-1)

        # 4. 应用 Dropout
        attn = self.dropout(attn)

        # 5. 将注意力权重与 V 相乘, 得到加权后的输出
        output = torch.matmul(attn, V)

        return output, attn
    
#多头注意力
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k

        # 定义 Q, K, V 的线性变换层
        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        # 最终的输出线性变换层
        self.W_O = nn.Linear(self.d_model, self.d_model)
        # 实例化缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout=config.dropout)

    def split_heads(self, x):
        """将最后一维 (d_model) 切分成 (n_heads, d_k)"""
        batch_size, seq_len, _ = x.size()
        # x: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # 交换 seq_len 和 n_heads 维度，方便并行计算: [batch_size, n_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """将多头的结果拼接起来"""
        # x: [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = x.size()
        # x: [batch_size, seq_len, n_heads, d_k] -> [batch_size, seq_len, d_model]
        return x.view(batch_size, seq_len, -1)

    def forward(self, Q, K, V, mask=None):
        # 1. 线性变换并切分为多头
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        # 2. 应用缩放点积注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 3. 拼接多头结果
        output = self.combine_heads(attn_output)

        # 4. 最终的线性变换
        output = self.W_O(output)
        return output, attn_weights
    
#位置编码
class PositionalEncoding(nn.Module):
    """生成位置编码，并添加到输入嵌入上"""
    def __init__(self, config):
        super().__init__()
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * -(math.log(10000.0) / config.d_model))

        # 计算正弦和余弦值并填充到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        # 将pe注册为buffer，确保它不参与模型参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 将位置编码加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return x
    
#前馈网络，全连接层，用于处理注意力模块的输出，进行非线性变化
class PositionWiseFFN(nn.Module):
    """位置前馈网络: FFN(x) = max(0, xW1 + b1)W2 + b2"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
    
#构建transformer
class TransformerBlock(nn.Module):
    """一个完整的Transformer编码器层"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = PositionWiseFFN(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # 1. 多头注意力 + 残差连接 + 层归一化
        # 注意：在自注意力中，Q、K、V都是同一个输入x
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        output = self.norm2(x + self.dropout2(ffn_output))
        return output
    
#调用
if __name__ == "__main__":
    # 1. 创建配置
    config = Config(d_model=512, n_heads=8, d_ff=2048)

    # 2. 实例化模型组件
    embedding = nn.Embedding(10000, config.d_model)  # 假设词汇表大小为10000
    pos_encoder = PositionalEncoding(config)
    transformer_block = TransformerBlock(config)

    # 3. 模拟一个batch数据: (batch_size, seq_len)
    batch_size, seq_len = 32, 50
    src = torch.randint(0, 10000, (batch_size, seq_len))

    # 4. 前向传播
    x = embedding(src)           # [32, 50, 512]
    x = pos_encoder(x)           # [32, 50, 512]
    output = transformer_block(x) # [32, 50, 512]

    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
