import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import sys

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from attention import MultiHeadAttention
from feed_forward import PositionWiseFFN
from layers import SublayerConnection
from positional_encoding import PositionalEncoding

class SimpleTransformer(nn.Module):
    """简单的Transformer模型（只有Encoder）"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2, max_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer层
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def _make_layer(self, d_model, n_heads, d_ff):
        # 创建单个Encoder层
        self_attn = MultiHeadAttention(d_model, n_heads)
        ffn = PositionWiseFFN(d_model, d_ff)
        
        return nn.ModuleDict({
            'self_attn': SublayerConnection(d_model, 0.1),
            'ffn': SublayerConnection(d_model, 0.1),
            'self_attn_module': self_attn,
            'ffn_module': ffn
        })
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有Transformer层
        for layer in self.layers:
            # 自注意力子层
            x = layer['self_attn'](x, lambda x: layer['self_attn_module'](x, x, x)[0])
            # 前馈子层
            x = layer['ffn'](x, layer['ffn_module'])
            
        # 输出预测
        output = self.output_layer(x)
        return output

class SimpleDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, texts, vocab, seq_length=20):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.texts) - self.seq_length
        
    def __getitem__(self, idx):
        # 获取输入序列和目标序列（语言建模任务）
        input_seq = self.texts[idx:idx + self.seq_length]
        target_seq = self.texts[idx + 1:idx + self.seq_length + 1]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)

def train():
    print("🚀 开始训练Transformer模型...")
    
    # 超参数
    vocab_size = 1000  # 简化词汇表大小
    d_model = 128
    n_heads = 4
    d_ff = 512
    n_layers = 2
    seq_length = 20
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # 创建简单数据（随机生成，用于测试）
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (1000,))  # 1000个token
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(data, None, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = SimpleTransformer(vocab_size, d_model, n_heads, d_ff, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练记录
    losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')
    print("📊 训练完成！损失曲线已保存到 results/training_loss.png")

if __name__ == "__main__":
    import math
    train()
