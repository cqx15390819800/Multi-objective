import numpy as np
K = ...  # 总目标数
M = ...  # 一次实验中要同时实验几组参数


def draw_weights(mu, sigma):
    weights = np.zeros((M, K))
    for j in range(K):
        # weights 的第 j 列，代表第 j 个目标的不同实验组的权重
        weights[:, j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(M,))
    return weights


def retain_top_weights(rewards, topN):
    # rewards[i][0] 是第 i 组实验的 reward（业务指标）
    # 按各组实验的业务指标从大到小排序
    rewards.sort(key=lambda x: x[0], reverse=True)

    top_weights = []
    for i in range(topN):
        # rewards[i][1]是第 i 组实验的 K 个权重
        top_weights.append(rewards[i][1])

    return np.asarray(top_weights)


# 参数初始化, mu 和 sigma 都是 K 维向量
mu = np.zeros(K)
sigma = np.ones(K) * init_sigma  # init_sigma 是 sigma 的初始值

for t in range(MaxRounds):  # MaxRounds 最多实验的轮数

    # 从 mu 和 sigma 指定的正态分布中，抽取 M 组超参
    # weights 的形状是 [M,K]，每行代表给一组实验的 K 个权重
    weights = draw_weights(mu, sigma)

    # do_experiments: 开 M 组小流量进行实验，返回 M 个实验结果
    # rewards 是 M 长的 list，每个元素是一个 tuple
    # rewards[i][0] 是第 i 组实验的 reward（业务指标）
    # rewards[i][1] 是第 i 组实验的 K 个权重
    rewards = do_experiments(weights)

    # 提取效果最好的 topN 组超参数
    # top_weights: [topN,K]
    top_weights = retain_top_weights(rewards, topN)

    # 用 topN 组超参数，更新 mu 和 sigma
    mu = top_weights.mean(axis=0)
    sigma = top_weights.std(axis=0)