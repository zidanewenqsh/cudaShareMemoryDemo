import numpy as np

# 初始化矩阵
A = np.arange(16).reshape(4, 4).astype(np.float32)
B = np.arange(16).reshape(4, 4).astype(np.float32)

# 计算矩阵乘法
C = np.dot(A, B)

# 打印结果
print(C)
