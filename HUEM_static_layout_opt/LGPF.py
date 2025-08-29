import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_distances(tensor_batch):
    """优化后的GPU距离计算"""
    # 输入应为 [N, 128] 的GPU张量
    assert tensor_batch.dim() == 2, "输入应为二维张量"

    # 使用矩阵运算优化距离计算 (复杂度O(N^2 + ND))
    sq_norm = torch.sum(tensor_batch ** 2, dim = 1, keepdim = True)  # [N, 1]
    dists = sq_norm + sq_norm.t() - 2 * torch.mm(tensor_batch, tensor_batch.t())
    dists = torch.sqrt(torch.clamp_min(dists, 1e-6))  # 避免负数

    # 屏蔽对角线 (自身距离)
    eye_mask = torch.eye(dists.size(0), device=device).bool()
    dists.masked_fill_(eye_mask, 1e6)
    return dists


def nearest_tensor(tensor_batch):
    """批量化最近邻计算"""
    with torch.no_grad():
        dists = calculate_distances(tensor_batch)
        return torch.min(dists, dim=1).values


def tensors_within_radius(tensor_batch, r):
    """向量化半径邻居搜索"""
    with torch.no_grad():
        dists = calculate_distances(tensor_batch)
        mask = dists < r

        # 使用张量操作获取索引和距离
        indices = [torch.nonzero(row).squeeze(1) for row in mask]
        distances = [dists[i][mask[i]] for i in range(mask.size(0))]

    return distances, indices


def newdata(da, T=0.3):
    x = torch.as_tensor(da, device=device, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        # 最近邻计算
        nearest = nearest_tensor(x)  # 形状 [N]
        G = nearest.mean()

        # 半径邻居搜索
        radius = 7 * G
        distances, indices = tensors_within_radius(x, radius)

        # 转换为填充张量
        max_neighbors = max(len(d) for d in distances) if distances else 0
        N = x.size(0)

        indices_pad = torch.full((N, max_neighbors), -1, device=device)
        dist_pad = torch.zeros(N, max_neighbors, device=device)
        valid_mask = torch.zeros(N, max_neighbors, dtype=torch.bool, device=device)

        for i, (idx, d) in enumerate(zip(indices, distances)):
            if len(idx) > 0:
                valid_len = min(len(idx), max_neighbors)
                indices_pad[i, :valid_len] = idx[:valid_len]
                dist_pad[i, :valid_len] = d[:valid_len]
                valid_mask[i, :valid_len] = True

        # 向量化计算
        batch_idx = torch.arange(N, device=device)[:, None]  # [N, 1]
        delta = x[indices_pad] - x[batch_idx]  # [N, K, 128]

        # 维度对齐修正
        nearest_expanded = nearest[:, None].expand(-1, max_neighbors)  # [N, K]
        dist_sq = dist_pad.pow(2)  # [N, K]

        # 作用力条件计算
        force_mask = (G * nearest_expanded > dist_sq) & valid_mask
        far_mask = (~force_mask) & valid_mask

        # 初始化力矩阵
        F = torch.zeros_like(delta)

        # 近距离作用力计算
        F[force_mask] = delta[force_mask] / 1.1

        # 远距离作用力维度修正
        if far_mask.any():
            # 获取有效元素
            valid_dist_sq = dist_sq[far_mask].unsqueeze(-1)  # [M, 1]
            nearest_valid = nearest_expanded[far_mask].unsqueeze(-1)  # [M, 1]
            delta_selected = delta[far_mask]  # [M, 128]

            # 维度对齐计算
            F_far = (nearest_valid * nearest_valid * delta_selected) / valid_dist_sq
            F[far_mask] = F_far.squeeze(-1)  # 移除最后增加的维度

        # 聚合作用力
        total_F = F.sum(dim=1) * T

    return x + total_F