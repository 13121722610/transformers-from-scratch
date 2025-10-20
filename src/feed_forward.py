import torch.nn as nn

class PositionWiseFFN(nn.Module):
    """位置级前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
