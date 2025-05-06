import torch
import torch.nn as nn
from typing import Callable,Union
from torch.optim import AdamW,SGD,LBFGS
import numpy as np
from tqdm import tqdm
from utils.sample import sample
import os
from torch.optim import Adam,SGD

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class get_Dataset():

    def __init__(self,device) -> None:
        self.device=device
    
    def sample_boundary(self):
        with torch.no_grad():
            data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
            boundary_internal = torch.from_numpy(data['boundary_internal']).float().to(device)
            boundary_external = torch.from_numpy(data['boundary_external']).float().to(device)
            return (boundary_internal,boundary_external)
    
    def sample_pde(self):
        data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        subdomains1 = torch.from_numpy(data['subdomains1']).float().to(device)
        subdomains2 = torch.from_numpy(data['subdomains2']).float().to(device)
        subdomains3 = torch.from_numpy(data['subdomains3']).float().to(device)
        return (subdomains1,subdomains2,subdomains3)
    
    def sample_interface(self):
        data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        interface12 = torch.from_numpy(data['interface12']).float().to(device)
        interface23 = torch.from_numpy(data['interface23']).float().to(device)
        return (interface12,interface23)


# 这里 x,t直接修改为x,y即可。仅为网络架构，采用的是MLP
class model(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=3, num_layers=5):
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
        # ini_shape = x.shape
        y = torch.stack([x, t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 随机初始化权重
# def xavier_init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)

# 损失计算

class get_loss(nn.Module):

    def _derivative(self, y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        for i in range(order):
            y = torch.autograd.grad(
                y, x, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True
            )[0]
        return y

    def pde_loss1(self,network,inputs,k):
        # 确保输入张量可以计算梯度
        x = inputs[:,0].clone().detach().requires_grad_(True)
        y = inputs[:,1].clone().detach().requires_grad_(True)
        
        T = network(x,y)[0]
        """ Physics-based loss function with Burgers equation """
        T_x = self._derivative(T, x, order=1)
        T_y = self._derivative(T, y, order=1)
        kT_x = k * T_x
        kT_y = k * T_y
        kT_xx = self._derivative(kT_x, x, order=1)
        kT_yy = self._derivative(kT_y, y, order=1)
        return (kT_xx + kT_yy).pow(2).mean()

    def pde_loss2(self,network,inputs,k):
        # 确保输入张量可以计算梯度
        x = inputs[:,0].clone().detach().requires_grad_(True)
        y = inputs[:,1].clone().detach().requires_grad_(True)
        
        T = network(x,y)[1]
        """ Physics-based loss function with Burgers equation """
        T_x = self._derivative(T, x, order=1)
        T_y = self._derivative(T, y, order=1)
        kT_x = k * T_x
        kT_y = k * T_y
        kT_xx = self._derivative(kT_x, x, order=1)
        kT_yy = self._derivative(kT_y, y, order=1)
        return (kT_xx + kT_yy).pow(2).mean()

    def pde_loss3(self,network,inputs,k):
        # 确保输入张量可以计算梯度
        x = inputs[:,0].clone().detach().requires_grad_(True)
        y = inputs[:,1].clone().detach().requires_grad_(True)
        
        T = network(x,y)[2]
        """ Physics-based loss function with Burgers equation """
        T_x = self._derivative(T, x, order=1)
        T_y = self._derivative(T, y, order=1)
        kT_x = k * T_x
        kT_y = k * T_y
        kT_xx = self._derivative(kT_x, x, order=1)
        kT_yy = self._derivative(kT_y, y, order=1)
        return (kT_xx + kT_yy).pow(2).mean()
    
    def boundary_loss1(self,network,inputs,value_b): 
        x_b = inputs[:,0]
        t_b = inputs[:,1]
        # x_b,t_b,value_b=inputs
        boundary_loss=torch.mean((network(x_b,t_b)[0]-value_b)**2)
        return boundary_loss

    def boundary_loss3(self,network,inputs,value_b): 
        x_b = inputs[:,0]
        t_b = inputs[:,1]
        # x_b,t_b,value_b=inputs
        boundary_loss=torch.mean((network(x_b,t_b)[2]-value_b)**2)
        return boundary_loss
    
    def get_continuity_loss1(self,network,inputs):
        # 假设inputs是一个形状为(N,2)的张量，其中第一列是x坐标，第二列是y坐标
        x_i = inputs[:,0]
        t_i = inputs[:,1]
        initial_loss=torch.mean((network(x_i,t_i)[0]-network(x_i,t_i)[1])**2)
        return initial_loss

    def get_continuity_loss2(self,network,inputs):
        # 假设inputs是一个形状为(N,2)的张量，其中第一列是x坐标，第二列是y坐标
        x_i = inputs[:,0]
        t_i = inputs[:,1]
        initial_loss=torch.mean((network(x_i,t_i)[1]-network(x_i,t_i)[2])**2)
        return initial_loss

def loss_1():
    grads = []
    optimizer1.zero_grad()
    interface_loss12 = loss.get_continuity_loss1(net, dataset.sample_interface()[0])
    interface_loss23 = loss.get_continuity_loss2(net, dataset.sample_interface()[1])
    boundary_lossin1 = loss.boundary_loss3(net, dataset.sample_boundary()[0], T1)
    #boundary_lossin2 = loss.boundary_loss3(net, dataset.sample_boundary()[0], T1)
    #boundary_lossin3 = loss.boundary_loss3(net, dataset.sample_boundary()[0], T1)
    boundary_lossout1 = loss.boundary_loss1(net, dataset.sample_boundary()[1], T2)
    #boundary_lossout2 = loss.boundary_loss1(net, dataset.sample_boundary()[1], T2)
    #boundary_lossout3 = loss.boundary_loss1(net, dataset.sample_boundary()[1], T2)
    
    PDE_loss1 = loss.pde_loss1(net, dataset.sample_pde()[0], k1)
    PDE_loss2 = loss.pde_loss2(net, dataset.sample_pde()[1], k2)
    PDE_loss3 = loss.pde_loss3(net, dataset.sample_pde()[2], k3)
    pde_loss = PDE_loss1 + PDE_loss2 + PDE_loss3
    boundary_loss_internal = boundary_lossin1
    boundary_loss_external = boundary_lossout1
    interface_loss = interface_loss12 + interface_loss23
    total_loss = pde_loss + boundary_loss_internal + boundary_loss_external + interface_loss
    total_loss.backward()
    
    #grad1 = get_gradient_vector(net)
    #grads = [grad1]
    #updated_grads = ConFIG_update(grads)
    #apply_gradient_vector(net, updated_grads)
    return total_loss

k1, k2, k3 = 1, 2, 3 # 热传导系数
T1, T2 = 10, 30 # 边界温度
net = model()
print(f"Total parameters: {net.count_parameters()}")
#net1.apply(xavier_init_weights)
net.to(device)
optimizer1 = LBFGS(net.parameters(),lr=1e-3)


dataset=get_Dataset(device=device)
loss=get_loss()

def train(device, log_file):
    best_loss = float('inf')
    
    # 创建保存目录
    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    k1, k2, k3 = 1, 2, 3 # 热传导系数
    T1, T2 = 10, 30 # 边界温度

    p_bar = tqdm(range(10000))   
    for i in p_bar:
        sample()
        total_loss_val = loss_1()
        optimizer1.step(loss_1)
        
        # 记录日志
        if i % 100 == 0:
            log_str = f"Iteration {i}: total_loss={total_loss_val.item():.6f}\n"
            with open(log_file, 'a') as f:
                f.write(log_str)
            

        
        # 保存最佳模型
        if total_loss_val < best_loss:
            best_loss = total_loss_val
            # 保存详细信息
            torch.save({
                'epoch': i,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': best_loss.item(),
            }, f'{save_dir}/best_model_single.pth')
            
            # 打印保存信息
            p_bar.set_description(f"Best model saved at iteration {i} with loss {best_loss.item():.6f}")

if __name__ == "__main__":
    seed = 19018
    # 创建日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 使用时间戳创建唯一的日志文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_log_{timestamp}.txt"
    torch.manual_seed(seed)
    np.random.seed(seed)
    train(device, log_file)
    
