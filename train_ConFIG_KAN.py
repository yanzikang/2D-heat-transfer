import torch
import torch.nn as nn
from typing import Callable,Union
from torch.optim import Adam,SGD
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from utils.sample import sample
from utils.abs_error import abs_error
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 这里 x,t直接修改为x,y即可。仅为网络架构，采用的是MLP
class model(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=4):
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


# 加载最佳模型并测试
def load_and_test():
    # 加载模型
    checkpoint = torch.load('checkpoints/best_model_single.pth')
    
    # 初始化网络
    net1 = model().to(device)
    net2 = model().to(device)
    net3 = model().to(device)
    
    # 加载模型参数
    net1.load_state_dict(checkpoint['net1_state_dict'])
    net2.load_state_dict(checkpoint['net2_state_dict'])
    net3.load_state_dict(checkpoint['net3_state_dict'])
    
    # 创建更精细的网格点
    n_points = 500
    x = np.linspace(-4, 4, n_points)
    y = np.linspace(-4, 4, n_points)
    X, Y = np.meshgrid(x, y)
    
    # 计算径向距离
    R = np.sqrt(X**2 + Y**2)
    
    # 创建温度数组
    T = np.zeros_like(R)
    
    # 转换为tensor进行预测
    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                r = R[i,j]
                if r < 1 or r > 4:  # r1=1, r4=4
                    T[i,j] = np.nan
                else:
                    # 分别传入x和y坐标作为网络的两个输入参数
                    x_coord = torch.tensor([X[i,j]], dtype=torch.float32).to(device)
                    y_coord = torch.tensor([Y[i,j]], dtype=torch.float32).to(device)
                    if 1 <= r <= 2:  # r1 <= r <= r2
                        T[i,j] = net1(x_coord, y_coord).cpu().numpy()
                    elif 2 < r <= 3:  # r2 < r <= r3
                        T[i,j] = net2(x_coord, y_coord).cpu().numpy()
                    elif 3 < r <= 4:  # r3 < r <= r4
                        T[i,j] = net3(x_coord, y_coord).cpu().numpy()
    
    return X, Y, T

def plot_results(X, Y, T):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(X, Y, T, cmap='jet', shading='auto')
    plt.colorbar(label='temperature')
    
    # 设置图形属性
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title('temperature distribution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # 只保存图形，不显示
    plt.savefig('predicted_temperature_field.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

if __name__ == "__main__":
    X, Y, T = load_and_test()
    print("绘制预测结果")
    plot_results(X, Y, T)
    print("绘制绝对误差")
    abs_error(checkpoint='checkpoints/best_model_single.pth')