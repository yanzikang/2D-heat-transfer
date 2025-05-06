import torch
import torch.nn as nn
from typing import Callable,Union
from torch.optim import Adam,SGD
import numpy as np
from tqdm import tqdm
import datetime  # 添加在文件开头的导入部分

from utils.sample import sample,sample_analytical
from utils.dataset import get_Dataset
from utils.model import model
from utils.loss import get_loss

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 随机初始化权重
def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# 损失计算

def train(device,log_file):
    # 在训练循环之前添加
    best_loss = float('inf')

    # 创建或清空日志文件
    with open(log_file, 'w') as f:
        f.write('Training Log\n')
        f.write('Epoch\tTotal Loss\tPDE Loss1\tPDE Loss2\tPDE Loss3\tInterface Loss12\tInterface Loss23\tBoundary Loss\tAnalytical Loss\n')

    net1 = model()
    net1.apply(xavier_init_weights)
    net1.to(device)
    optimizer1 = Adam(net1.parameters(),lr=1e-4)

    net2 = model()
    net2.apply(xavier_init_weights)
    net2.to(device)
    optimizer2 = Adam(net2.parameters(),lr=1e-4)

    net3 = model()
    net3.apply(xavier_init_weights)
    net3.to(device)
    optimizer3 = Adam(net3.parameters(),lr=1e-4)

    dataset=get_Dataset(device=device)
    loss=get_loss()

    k1, k2, k3 = 1, 2, 3 # 热传导系数
    T1, T2 = 10, 30 # 边界温度

    p_bar=tqdm(range(100000))   
    for i in p_bar:
        sample()
        grads = []
        losses = []
        
        # 清除所有梯度
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        
        # 计算所有损失
        interface_loss12 = loss.get_continuity_loss(net1, net2, dataset.sample_interface()[0])
        interface_loss23 = loss.get_continuity_loss(net2, net3, dataset.sample_interface()[1])
        
        boundary_lossin1 = loss.boundary_loss(net1, dataset.sample_boundary()[0], T1)
        boundary_lossin2 = loss.boundary_loss(net2, dataset.sample_boundary()[0], T1)
        boundary_lossin3 = loss.boundary_loss(net3, dataset.sample_boundary()[0], T1)
        
        boundary_lossout1 = loss.boundary_loss(net1, dataset.sample_boundary()[1], T2)
        boundary_lossout2 = loss.boundary_loss(net2, dataset.sample_boundary()[1], T2)
        boundary_lossout3 = loss.boundary_loss(net3, dataset.sample_boundary()[1], T2)
        
        PDE_loss1 = loss.pde_loss(net1, dataset.sample_pde()[0], k1)
        PDE_loss2 = loss.pde_loss(net2, dataset.sample_pde()[1], k2)
        PDE_loss3 = loss.pde_loss(net3, dataset.sample_pde()[2], k3)

        # gt_loss1 = loss.analytical_loss(net1,dataset.get_analytical_solutions()[0],dataset.get_analytical_solutions()[1])
        # gt_loss2 = loss.analytical_loss(net2,dataset.get_analytical_solutions()[2],dataset.get_analytical_solutions()[3])
        # gt_loss3 = loss.analytical_loss(net3,dataset.get_analytical_solutions()[4],dataset.get_analytical_solutions()[5])
        
        # 计算总损失
        total_loss1 = interface_loss12 + boundary_lossin1 + boundary_lossout1 + PDE_loss1
        total_loss2 = interface_loss12 + interface_loss23 + boundary_lossin2 + boundary_lossout2 + PDE_loss2
        total_loss3 = interface_loss23 + boundary_lossin3 + boundary_lossout3 + PDE_loss3
        # total_loss4 = gt_loss1 + gt_loss2 + gt_loss3
        # 一次性计算所有梯度
        total_loss = total_loss1 + total_loss2 + total_loss3 
        total_loss.backward()
        
        # 收集梯度
        grad1 = get_gradient_vector(net1)
        grad2 = get_gradient_vector(net2)
        grad3 = get_gradient_vector(net3)
        grads = [grad1, grad2, grad3]
        
        # 更新梯度
        updated_grads = ConFIG_update(grads)
        apply_gradient_vector(net1, updated_grads)
        apply_gradient_vector(net2, updated_grads)
        apply_gradient_vector(net3, updated_grads)
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        if i % 100 == 0:
            total_loss_val = total_loss.item()
            p_bar.set_description(f'Loss: {total_loss_val:.6f}')
            
            # 保存最佳模型
            if total_loss_val < best_loss:
                best_loss = total_loss_val
                torch.save({
                    'epoch': i,
                    'net1_state_dict': net1.state_dict(),
                    'net2_state_dict': net2.state_dict(),
                    'net3_state_dict': net3.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'optimizer3_state_dict': optimizer3.state_dict(),
                    'loss': best_loss,
                }, 'checkpoints/best_model.pth')
                
            # 计算边界损失总和
            boundary_loss_total = (boundary_lossin1 + boundary_lossin2 + boundary_lossin3 + 
                                boundary_lossout1 + boundary_lossout2 + boundary_lossout3).item()
            # analytical_loss_total = (gt_loss1 + gt_loss2 + gt_loss3).item()
            
            with open(log_file, 'a') as f:
                f.write(f'{i}\t{total_loss_val:.6f}\t{PDE_loss1.item():.6f}\t{PDE_loss2.item():.6f}\t'
                    f'{PDE_loss3.item():.6f}\t{interface_loss12.item():.6f}\t{interface_loss23.item():.6f}\t'
                    f'{boundary_loss_total:.6f}\n')

if __name__ == "__main__":
    sample_analytical()
    seed = 19018
    device = "cuda:0"
    
    # 设置完整的随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 生成带有时间戳的日志文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{current_time}.txt"
    
    # 确保logs目录存在
    import os
    os.makedirs("logs", exist_ok=True)
    
    train(device,log_file)
