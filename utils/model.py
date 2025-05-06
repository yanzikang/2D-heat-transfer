import torch.nn as nn
import torch
# from kan import KAN
# model = KAN([2, 20,20,20,20, 1])
class model(nn.Module):
    
    def __init__(self, input_dim=3, hidden_dim=20, output_dim=1, num_layers=4):
        super(model, self).__init__()
        
        # 初始层
        self.ini_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 中间层
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
        # 输出层
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        ini_shape = x.shape
        y = torch.stack([x, t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y.view(ini_shape)