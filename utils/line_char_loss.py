import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(txt_file_path1, txt_file_path2):
    """
    读取两个txt文件并绘制最后20000轮的损失曲线对比图
    参数:
        txt_file_path1: 第一个txt文件的路径
        txt_file_path2: 第二个txt文件的路径
    """
    # 读取第一个文件
    with open(txt_file_path1, 'r') as f:
        header = f.readline().strip().split()
        data1 = np.loadtxt(txt_file_path1, skiprows=1)
        # 只取最后20000轮的数据
        if len(data1) > 20000:
            data1 = data1[-20000:]
        x1 = data1[:, 0]
        y1 = data1[:, 1]

    # 读取第二个文件
    with open(txt_file_path2, 'r') as f:
        header = f.readline().strip().split()
        data2 = np.loadtxt(txt_file_path2, skiprows=1)
        # 只取最后20000轮的数据
        if len(data2) > 20000:
            data2 = data2[-20000:]
        x2 = data2[:, 0]
        y2 = data2[:, 1]

    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制两条损失曲线
    plt.plot(x1, y1, label='Loss 1', linewidth=2)
    plt.plot(x2, y2, label='Loss 2', linewidth=2)
    
    # 设置图形属性
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves Comparison (Last 20000 Iterations)')
    plt.legend()
    plt.grid(True)
    
    # 保存图形
    plt.savefig('loss_curves_comparison_last_20000.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 使用示例
    txt_file_path1 = "/DATA/yanzikang/PINN/logs/training_log_20250310_153613.txt"
    txt_file_path2 = "/DATA/yanzikang/PINN/logs/training_log_20250310_160442.txt"  # 替换为第二个文件的实际路径
    plot_loss_curves(txt_file_path1, txt_file_path2)
