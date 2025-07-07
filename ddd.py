import torch


def radial_bias_sampling_groups(image_patches, n_group=2, new_n=64, alpha=4.0):
    """
    Radial Bias Sampling: generate n_group sampled patch sets per image.

    Args:
        image_patches (Tensor): (bs, n, d) — input patches
        n_group (int): number of views (groups) to sample
        new_n (int): number of patches to sample per view
        alpha (float): exponential decay rate (controls bias sharpness)

    Returns:
        List[Tensor]: list of n_group tensors, each of shape (bs, new_n, d)
    """
    bs, n, d = image_patches.shape
    side_len = int(n ** 0.5)
    assert side_len ** 2 == n, "n must be a perfect square (e.g., 196=14x14)"
    device = image_patches.device

    # === Step 1: 2D coordinates of each patch (n, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(side_len, device=device),
        torch.arange(side_len, device=device),
        indexing='ij'
    )
    coords = torch.stack([grid_y, grid_x], dim=-1).reshape(n, 2).float()  # (n, 2)

    # === Step 2: Random center indices for each view & batch: (bs, n_group)
    center_indices = torch.randint(0, n, (bs, n_group), device=device)  # (bs, n_group)
    center_coords = coords[center_indices]  # (bs, n_group, 2)

    # === Step 3: Expand coords for broadcasting
    coords_exp = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, n, 2)
    center_coords_exp = center_coords.unsqueeze(2)  # (bs, n_group, 1, 2)

    # === Step 4: Compute distances: (bs, n_group, n)
    dists = torch.norm(coords_exp - center_coords_exp, dim=-1)  # (bs, n_group, n)

    # === Step 5: Compute sampling probabilities
    weights = torch.exp(-alpha * dists)  # (bs, n_group, n)
    probs = weights / weights.sum(dim=-1, keepdim=True)  # (bs, n_group, n)

    # === Step 6: Sample indices: (bs, n_group, new_n)
    probs_2d = probs.reshape(bs * n_group, n)  # flatten batch × group
    sampled_indices_2d = torch.multinomial(probs_2d, new_n, replacement=False)  # (bs, n_group, new_n)
    sampled_indices = sampled_indices_2d.view(bs, n_group, new_n)  # (bs, n_group, new_n)

    # === Step 7: Gather patches
    # Prepare gather index: (bs, n_group, new_n, d)
    idx_expanded = sampled_indices.unsqueeze(-1).expand(-1, -1, -1, d)  # (bs, n_group, new_n, d)
    image_patches_exp = image_patches.unsqueeze(1).expand(-1, n_group, -1, -1)  # (bs, n_group, n, d)

    sampled = torch.gather(image_patches_exp, dim=2, index=idx_expanded)  # (bs, n_group, new_n, d)

    # === Step 8: Return as list of (bs, new_n, d)
    return [sampled[:, i] for i in range(n_group)]  # list of (bs, new_n, d)


if __name__ == '__main__':
    bs, n, d = 8, 196, 512  # 14x14 patches
    image_patches = torch.randn(bs, n, d)
    views = radial_bias_sampling_groups(image_patches, n_group=2, new_n=64, alpha=5.0)

    pass