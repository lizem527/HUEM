import LGPF
import COV
import pandas as pd
import numpy as np
import time
# import cupy as cp

if __name__ == '__main__':

    # 生成测试数据
    np_data = np.random.randn(1000, 128).astype(np.float32)

    # 开始计时
    time_start = time.time()

    cov_mat, scale_mat, rotation_mat = COV.covariance_eigen_decomposition(np_data)

    # 数据转化
    A = COV.transform_data(cov_mat, scale_mat, rotation_mat)
    transform_data = np_data @ A
    # 执行LPGF位移
    result = LGPF.newdata(transform_data)

    # 开始计时
    time_end = time.time()
    # 计算执行时间
    time_elapsed = time_end - time_start
    print('运行时间：', time_elapsed, '秒')

    print(result)













