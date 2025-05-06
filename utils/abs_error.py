import torch
from typing import Callable,Union
import numpy as np
from scipy.stats.qmc import LatinHypercube
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.sample import sample
print(f"Using device: {device}")
import matplotlib.pyplot as plt

# 解析解结果
def T1_analytical(x1, x2, T1_val, A, r1, r2):
    """
    计算区域Ω1的温度分布
    T1(x1, x2) = T1 + (A-T1)/ln(r2/r1) * ln(r/r1)
    """
    r = np.sqrt(x1**2 + x2**2)  # 计算径向距离
    return T1_val + (A - T1_val) / np.log(r2/r1) * np.log(r/r1)

def T2_analytical(x1, x2, A, B, r2, r3):
    """
    计算区域Ω2的温度分布
    T2(x1, x2) = A + (A-B)/ln(r2/r3) * ln(r/r2)
    """
    r = np.sqrt(x1**2 + x2**2)  # 计算径向距离
    return A + (A - B) / np.log(r2/r3) * np.log(r/r2)

def T3_analytical(x1, x2, T2_val, B, r3, r4):
    """
    计算区域Ω3的温度分布
    T3(x1, x2) = T2 + (B-T2)/ln(r3/r4) * ln(r/r4)
    """
    r = np.sqrt(x1**2 + x2**2)  # 计算径向距离
    return T2_val + (B - T2_val) / np.log(r3/r4) * np.log(r/r4)

def calculate_A(T1, T2, k1, k2, k3, r1, r2, r3, r4):
    """
    计算参数A的值
    A = (T1*k1*(k3*ln(r3/r2) + k2*ln(r4/r3)) + T2*k2*k3*ln(r2/r1)) / 
        (k1*k3*ln(r3/r2) + k1*k2*ln(r4/r3) + k2*k3*ln(r2/r1))
    """
    # 分子部分
    numerator = (T1 * k1 * (k3 * np.log(r3/r2) + k2 * np.log(r4/r3)) + 
                 T2 * k2 * k3 * np.log(r2/r1))
    
    # 分母部分
    denominator = (k1 * k3 * np.log(r3/r2) + 
                  k1 * k2 * np.log(r4/r3) + 
                  k2 * k3 * np.log(r2/r1))
    
    return numerator / denominator

def calculate_B(T1, T2, k1, k2, k3, r1, r2, r3, r4):
    """
    计算参数B的值
    B = (T1*k1*k2*ln(r4/r3) + T2*k3*(k1*ln(r3/r2) + k2*ln(r2/r1))) /
        (k1*k2*ln(r4/r3) + k1*k3*ln(r3/r2) + k2*k3*ln(r2/r1))
    """
    # 分子部分
    numerator = (T1 * k1 * k2 * np.log(r4/r3) + 
                 T2 * k3 * (k1 * np.log(r3/r2) + k2 * np.log(r2/r1)))
    
    # 分母部分
    denominator = (k1 * k2 * np.log(r4/r3) + 
                  k1 * k3 * np.log(r3/r2) + 
                  k2 * k3 * np.log(r2/r1))
    
    return numerator / denominator


# 设置参数
k1, k2, k3 = 1, 2, 3
r1, r2, r3, r4 = 1, 2, 3, 4
T1_val, T2_val = 10, 30

class BurgersDataset():
    
    def __init__(self,
                 x_start:float=-1.0,x_end:float=1.0,ini_boundary:Callable=lambda x: -1*torch.sin(np.pi*x),simulation_time:float=1.0,
                 n_pde:int=2000,n_initial:int=50,n_boundary:int=50,device:Union[str,torch.device]="cpu",
                 nu=0.01/np.pi,
                 seed:int=21339,) -> None:
        self.ini_boundary=ini_boundary
        self.x_start=x_start
        self.x_end=x_end
        self.n_pde=n_pde
        self.n_boundary=n_boundary
        self.n_initial=n_initial
        self.nu=nu
        self.simulation_time=simulation_time
        self.device=device
        self.random_engine_initial_boundary=LatinHypercube(d=1,seed=seed)
        self.random_engine_pde=LatinHypercube(d=2,seed=seed)
    
    
    def sample_boundary(self):
        with torch.no_grad():
            data = np.load("sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
            boundary_internal = torch.from_numpy(data['boundary_internal']).float().to(device)
            boundary_external = torch.from_numpy(data['boundary_external']).float().to(device)
            return (boundary_internal,boundary_external)
    
    def sample_pde(self):
        data = np.load("sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        subdomains1 = torch.from_numpy(data['subdomains1']).float().to(device)
        subdomains2 = torch.from_numpy(data['subdomains2']).float().to(device)
        subdomains3 = torch.from_numpy(data['subdomains3']).float().to(device)
        return (subdomains1,subdomains2,subdomains3)
    
    def sample_interface(self):
        data = np.load("sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        interface12 = torch.from_numpy(data['interface12']).float().to(device)
        interface23 = torch.from_numpy(data['interface23']).float().to(device)
        return (interface12,interface23)



import torch.nn as nn

# 这里 x,t直接修改为x,y即可。仅为网络架构，采用的是MLP
class BurgersNet(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=4):
        super(BurgersNet, self).__init__()
        
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

# 随机初始化权重
def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# 损失计算

class BurgersLoss(nn.Module):

    def _derivative(self, y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        for i in range(order):
            y = torch.autograd.grad(
                y, x, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True
            )[0]
        return y

    def pde_loss(self,network,inputs,k):
        # 确保输入张量可以计算梯度
        x = inputs[:,0].clone().detach().requires_grad_(True)
        y = inputs[:,1].clone().detach().requires_grad_(True)
        
        T = network(x,y)
        """ Physics-based loss function with Burgers equation """
        T_x = self._derivative(T, x, order=1)
        T_y = self._derivative(T, y, order=1)
        kT_x = k * T_x
        kT_y = k * T_y
        kT_xx = self._derivative(kT_x, x, order=1)
        kT_yy = self._derivative(kT_y, y, order=1)
        return (kT_xx + kT_yy).pow(2).mean()
    
    def boundary_loss(self,network,inputs,value_b): 
        x_b = inputs[:,0]
        t_b = inputs[:,1]
        # x_b,t_b,value_b=inputs
        boundary_loss=torch.mean((network(x_b,t_b)-value_b)**2)
        return boundary_loss
    
    def get_continuity_loss(self,network1,network2,inputs):
        # 假设inputs是一个形状为(N,2)的张量，其中第一列是x坐标，第二列是y坐标
        x_i = inputs[:,0]
        t_i = inputs[:,1]
        initial_loss=torch.mean((network1(x_i,t_i)-network2(x_i,t_i))**2)
        return initial_loss
    
from torch.optim import Adam,SGD
from tqdm import tqdm

seed=19018
device="cuda:0"
log_file = "training_log.txt"

torch.manual_seed(seed)
np.random.seed(seed)
net1 = BurgersNet()
net1.apply(xavier_init_weights)
net1.to(device)
optimizer1 = Adam(net1.parameters(),lr=1e-4)

net2 = BurgersNet()
net2.apply(xavier_init_weights)
net2.to(device)
optimizer2 = Adam(net2.parameters(),lr=1e-4)

net3 = BurgersNet()
net3.apply(xavier_init_weights)
net3.to(device)
optimizer3 = Adam(net3.parameters(),lr=1e-4)



dataset=BurgersDataset(device=device,seed=seed)
loss=BurgersLoss()
#tester=Tester()
k1, k2, k3 = 1, 2, 3 # 热传导系数
T1, T2 = 10, 30 # 边界温度



from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector



# 添加测试函数
def test_model(net1, net2, net3, test_points):
    """
    对训练好的模型进行测试
    test_points: 形状为(N, 2)的张量，包含测试点的x和y坐标
    """
    net1.eval()
    net2.eval()
    net3.eval()
    
    with torch.no_grad():
        # 根据x坐标确定使用哪个网络
        results = []
        for point in test_points:
            x, y = point[0], point[1]
            if x <= -1/3:
                temp = net1(x.unsqueeze(0), y.unsqueeze(0))
            elif x <= 1/3:
                temp = net2(x.unsqueeze(0), y.unsqueeze(0))
            else:
                temp = net3(x.unsqueeze(0), y.unsqueeze(0))
            results.append(temp.item())
    
    return torch.tensor(results)

# 加载最佳模型并测试
def load_and_test(checkpoint):
    # 加载模型
    checkpoint = torch.load(checkpoint)
    
    # 初始化网络
    net1 = BurgersNet().to(device)
    net2 = BurgersNet().to(device)
    net3 = BurgersNet().to(device)
    
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
    total_error = 0  # 添加总误差计数器
    valid_points = 0  # 添加有效点计数器
    
    # 转换为tensor进行预测
    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                r = R[i,j]
                if r < 1 or r > 4:  # r1=1, r4=4
                    T[i,j] = np.nan
                else:
                    valid_points += 1  # 增加有效点计数
                    # 分别传入x和y坐标作为网络的两个输入参数
                    A = calculate_A(T1_val, T2_val, k1, k2, k3, r1, r2, r3, r4)
                    B = calculate_B(T1_val, T2_val, k1, k2, k3, r1, r2, r3, r4)

                    x_coord = torch.tensor([X[i,j]], dtype=torch.float32).to(device)
                    y_coord = torch.tensor([Y[i,j]], dtype=torch.float32).to(device)
                    if 1 <= r <= 2:  # r1 <= r <= r2
                        error = abs(net1(x_coord, y_coord).cpu().numpy()-T1_analytical(X[i,j], Y[i,j], T1_val, A, r1, r2))
                    elif 2 < r <= 3:  # r2 < r <= r3
                        error = abs(net2(x_coord, y_coord).cpu().numpy()-T2_analytical(X[i,j], Y[i,j], A, B, r2, r3))
                    elif 3 < r <= 4:  # r3 < r <= r4
                        error = abs(net3(x_coord, y_coord).cpu().numpy()-T3_analytical(X[i,j], Y[i,j], T2_val, B, r3, r4))
                    T[i,j] = error
                    total_error += error  # 累加误差
    
    avg_error = total_error / valid_points  # 计算平均误差
    print(f"总绝对误差: {total_error}")
    print(f"平均绝对误差: {avg_error}")
    
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
    plt.savefig('predicted_temperature_field_abs_error.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

def abs_error(checkpoint):
    X, Y, T = load_and_test(checkpoint)
    plot_results(X, Y, T)

if __name__ == "__main__":
    X, Y, T = load_and_test(checkpoint)
    plot_results(X, Y, T)

 # 其实预测出来的结果相较于之前的结果来说已经又了很大的进步了，但是最终的效果依旧不是很好，需要进一步的优化
 # 在损失中加入真实的温度分布，可以进一步的优化结果？
 # 上面的结论在GPT-PINN中有提到对于边界条件缺失的情况，可以采用时间数据作为补充，取少量的实验数据，试一试