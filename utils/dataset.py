import torch
import numpy as np

class get_Dataset():

    def __init__(self,device) -> None:
        self.device=device
    
    def sample_boundary(self):
        with torch.no_grad():
            data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
            boundary_internal = torch.from_numpy(data['boundary_internal']).float().to(self.device)
            boundary_external = torch.from_numpy(data['boundary_external']).float().to(self.device)
            return (boundary_internal,boundary_external)
    
    def sample_pde(self):
        data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        subdomains1 = torch.from_numpy(data['subdomains1']).float().to(self.device)
        subdomains2 = torch.from_numpy(data['subdomains2']).float().to(self.device)
        subdomains3 = torch.from_numpy(data['subdomains3']).float().to(self.device)
        return (subdomains1,subdomains2,subdomains3)
    
    def sample_interface(self):
        data = np.load("data/sampling_points.npy", allow_pickle=True).item()  # 将数组转换为字典
        interface12 = torch.from_numpy(data['interface12']).float().to(self.device)
        interface23 = torch.from_numpy(data['interface23']).float().to(self.device)
        return (interface12,interface23)
    
    def get_analytical_solutions(self):
        data = np.load("data/analytical_solutions.npy", allow_pickle=True).item()  # 将数组转换为字典
        s1 = torch.from_numpy(data['subdomains1_points']).float().to(self.device)
        s1_T = torch.from_numpy(data['subdomains1_values']).float().to(self.device)
        s2 = torch.from_numpy(data['subdomains2_points']).float().to(self.device)
        s2_T = torch.from_numpy(data['subdomains2_values']).float().to(self.device)
        s3 = torch.from_numpy(data['subdomains3_points']).float().to(self.device)
        s3_T = torch.from_numpy(data['subdomains3_values']).float().to(self.device)

        return(s1,s1_T,s2,s2_T,s3,s3_T)
        
