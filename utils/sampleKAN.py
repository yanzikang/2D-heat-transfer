# 构建模型用于采样
"""
边界点采样：
最内侧边界点:200 r1=1 # boundary_internal = sample_boundary_points(200, 1)
最外侧边界点:200 r4=4 # boundary_external = sample_boundary_points(200, 4)
交界面采样：
1,2交界面:200 r2=2 # interface12 = sample_interface_points(200, 2)
2,3交界面:200 r3=3 # interface23 = sample_interface_points(200, 3)
介质点采样：
介质1:408  r1<r<r2 # subdomains1 = sample_annulus_points(408, 1, 2)
介质2:692  r2<r<r3 # subdomains2 = sample_annulus_points(692, 2, 3)
介质3:964  r3<r<r4 # subdomains3 = sample_annulus_points(964, 3, 4)
"""
def sample_boundary_points(num_points, radius):
    """在指定半径的圆周上随机采样点
    
    Args:
        num_points (int): 采样点的数量
        radius (float): 圆周半径
        
    Returns:
        tuple: 包含两个numpy数组，分别表示x坐标和y坐标
    """
    import numpy as np
    np.random.seed(42)  # 添加随机数种子
    
    # 生成随机分布的角度
    theta = np.random.uniform(0, 2*np.pi, num_points)
    
    # 计算x和y坐标
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    return x, y

def sample_interface_points(num_points, radius):
    """在指定半径的圆周上随机采样点
    
    Args:
        num_points (int): 采样点的数量
        radius (float): 圆周半径
        
    Returns:
        tuple: 包含两个numpy数组，分别表示x坐标和y坐标
    """
    import numpy as np
    np.random.seed(42)  # 添加随机数种子
    
    # 生成随机分布的角度
    theta = np.random.uniform(0, 2*np.pi, num_points)
    
    # 计算x和y坐标
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    return x, y

def sample_annulus_points(num_points, r_inner, r_outer):
    """在空心圆（环形）区域内随机采样点，不包含边界
    
    Args:
        num_points (int): 采样点的数量
        r_inner (float): 内圆半径
        r_outer (float): 外圆半径
        
    Returns:
        tuple: 包含两个numpy数组，分别表示x坐标和y坐标
    """
    import numpy as np
    np.random.seed(42)  # 添加随机数种子
    
    # 生成随机角度 [0, 2π)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    
    # 在半径的平方上进行均匀采样，添加一个很小的偏移量以避免边界
    epsilon = 1e-10  # 定义一个很小的偏移量
    r = np.sqrt(np.random.uniform((r_inner + epsilon)**2, (r_outer - epsilon)**2, num_points))
    
    # 计算x和y坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y


def plt_sample():
    import matplotlib.pyplot as plt

    # 创建新的图形
    plt.figure(figsize=(10, 10))

    # 绘制所有采样点
    plt.scatter(boundary_internal[0], boundary_internal[1], c='red', label='Inner Boundary (r=1)', s=20)
    plt.scatter(boundary_external[0], boundary_external[1], c='blue', label='Outer Boundary (r=4)', s=20)
    plt.scatter(interface12[0], interface12[1], c='green', label='Interface 1-2 (r=2)', s=20)
    plt.scatter(interface23[0], interface23[1], c='purple', label='Interface 2-3 (r=3)', s=20)
    plt.scatter(subdomains1[0], subdomains1[1], c='yellow', label='Subdomain 1', alpha=0.5, s=20)
    plt.scatter(subdomains2[0], subdomains2[1], c='orange', label='Subdomain 2', alpha=0.5, s=20)
    plt.scatter(subdomains3[0], subdomains3[1], c='cyan', label='Subdomain 3', alpha=0.5, s=20)

    # 设置图形属性
    plt.axis('equal')  # 保持横纵比例相等
    plt.grid(True)     # 显示网格
    plt.legend()       # 显示图例
    plt.title('sample result')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.show()

def sample():
    boundary_internal = sample_boundary_points(200, 1)
    boundary_external = sample_boundary_points(200, 4)
    interface12 = sample_interface_points(200, 2)
    interface23 = sample_interface_points(200, 3)
    subdomains1 = sample_annulus_points(420, 1, 2)
    subdomains2 = sample_annulus_points(690, 2, 3)
    subdomains3 = sample_annulus_points(960, 3, 4)

    # 将所有采样点保存为单个npy文件
    import numpy as np

    # 将每组采样点转换为形状为(n, 2)的数组
    sampling_data = {
        'boundary_internal': np.column_stack(boundary_internal),
        'boundary_external': np.column_stack(boundary_external),
        'interface12': np.column_stack(interface12),
        'interface23': np.column_stack(interface23),
        'subdomains1': np.column_stack(subdomains1),
        'subdomains2': np.column_stack(subdomains2),
        'subdomains3': np.column_stack(subdomains3)
    }

    # 保存为单个npy文件
    np.save('data/sampling_points_KAN.npy', sampling_data)

# 由于边界条件的缺失，现在补充一组损失，为真实值损失，该损失占比非常的大，该方法用于数据驱动，
"""
从损失上来看是比较低的，但是从实际的图片上来看并不符合实际要求，故引入真实值损失

"""
# 用于复现解析解
import numpy as np
import matplotlib.pyplot as plt
def sample_analytical():
    def T1(x1, x2, T1_val, A, r1, r2):
        """
        计算区域Ω1的温度分布
        T1(x1, x2) = T1 + (A-T1)/ln(r2/r1) * ln(r/r1)
        """
        r = np.sqrt(x1**2 + x2**2)  # 计算径向距离
        return T1_val + (A - T1_val) / np.log(r2/r1) * np.log(r/r1)

    def T2(x1, x2, A, B, r2, r3):
        """
        计算区域Ω2的温度分布
        T2(x1, x2) = A + (A-B)/ln(r2/r3) * ln(r/r2)
        """
        r = np.sqrt(x1**2 + x2**2)  # 计算径向距离
        return A + (A - B) / np.log(r2/r3) * np.log(r/r2)

    def T3(x1, x2, T2_val, B, r3, r4):
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

    # 创建更精细的网格点
    n_points = 500  # 从200增加到500
    x = np.linspace(-4, 4, n_points)
    y = np.linspace(-4, 4, n_points)
    X, Y = np.meshgrid(x, y)

    # 计算每个点的径向距离
    R = np.sqrt(X**2 + Y**2)

    # 创建温度数组
    T = np.zeros_like(R)

    # 设置参数
    k1, k2, k3 = 1, 2, 3
    r1, r2, r3, r4 = 1, 2, 3, 4
    T1_val, T2_val = 10, 30

    # 计算参数A和B
    A = calculate_A(T1_val, T2_val, k1, k2, k3, r1, r2, r3, r4)
    B = calculate_B(T1_val, T2_val, k1, k2, k3, r1, r2, r3, r4)

    # 计算温度分布
    for i in range(n_points):
        for j in range(n_points):
            r = R[i,j]
            if r < r1 or r > r4:
                T[i,j] = np.nan  # 设置范围外的值为NaN
            elif r1 <= r <= r2:
                T[i,j] = T1(X[i,j], Y[i,j], T1_val, A, r1, r2)
            elif r2 < r <= r3:
                T[i,j] = T2(X[i,j], Y[i,j], A, B, r2, r3)
            elif r3 < r <= r4:
                T[i,j] = T3(X[i,j], Y[i,j], T2_val, B, r3, r4)

    # 每个区域的解析解
    subdomains1 = sample_annulus_points(420, 1, 2)
    T_s1 = T1(subdomains1[0], subdomains1[1], T1_val, A, r1, r2)

    subdomains2 = sample_annulus_points(690, 2, 3)
    T_s2 = T2(subdomains2[0], subdomains2[1], A, B, r2, r3)

    subdomains3 = sample_annulus_points(960, 3, 4)
    T_s3 = T3(subdomains3[0], subdomains3[1], T2_val, B, r3, r4)

    # 将采样点和对应的解析解保存为npy文件
    analytical_data = {
        'subdomains1_points': np.column_stack(subdomains1),
        'subdomains1_values': T_s1,
        'subdomains2_points': np.column_stack(subdomains2),
        'subdomains2_values': T_s2,
        'subdomains3_points': np.column_stack(subdomains3),
        'subdomains3_values': T_s3
    }

    # 保存为npy文件
    np.save('data/analytical_solutions.npy', analytical_data)
