# 用于复现解析解
import numpy as np
import matplotlib.pyplot as plt

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

# 创建图形
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

# 显示图形
plt.show()
