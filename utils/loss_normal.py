import torch
import torch.nn as nn

class get_loss(nn.Module):

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

    def analytical_loss(self,network,inputs,gt):
        x = inputs[:,0]
        y = inputs[:,1]
        analytical_loss=torch.mean((network(x,y)-gt)**2)
        return analytical_loss
    