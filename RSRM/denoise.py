import numpy as np
from scipy.signal import savgol_filter

# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 生成一个没有噪声的9x25数据集
# 这里我们使用一个简单的线性趋势作为基础信号
base_data = np.linspace(0, 2 * np.pi, 10).reshape((2, 5))

# 添加高斯噪声
# 假设噪声的标准差为0.1（可以根据需要调整）
noise_std = 0.1
noisy_data = base_data + np.random.normal(0, noise_std, base_data.shape)

# 定义滤波器参数
window_size = 5  # 滤波器窗口大小（必须为奇数）
poly_order = 2    # 多项式的阶数

# 应用Savitzky-Golay滤波器于每一行
smoothed_data = np.array([savgol_filter(row, window_size, poly_order) for row in noisy_data])

# 输出原始噪声数据和平滑后的数据
print("Original noisy data (first 5 rows):")
print(noisy_data[:5, :])
print("\nSmoothed data (first 5 rows):")
print(smoothed_data[:5, :])