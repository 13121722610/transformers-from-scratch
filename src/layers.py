import torch.nn as nn

class SublayerConnection(nn.Module):
    """残差连接 + LayerNorm"""
    
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # 残差连接: x + dropout(sublayer(norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))
