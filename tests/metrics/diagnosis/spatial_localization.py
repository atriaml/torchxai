import torch
from torch import Tensor


def _reduce_groups(
    attr: Tensor,         # [N]
    mask: Tensor,         # [N]
    centroids: Tensor,    # [N, D]
):
    """
    Reduce raw features → grouped features.

    Returns:
        group_attr: [G]
        group_centroids: [G, D]
    """
    n_groups = int(mask.max().item()) + 1

    # --- attribution sum ---
    group_attr = torch.zeros(n_groups, device=attr.device)
    group_attr.scatter_add_(0, mask.long(), attr)

    # --- centroid mean ---
    D = centroids.shape[-1]

    group_centroids = torch.zeros(n_groups, D, device=attr.device)
    counts = torch.zeros(n_groups, device=attr.device)

    group_centroids.scatter_add_(
        0,
        mask.unsqueeze(-1).expand(-1, D),
        centroids
    )
    counts.scatter_add_(0, mask, torch.ones_like(attr))

    counts = counts.clamp(min=1.0)
    group_centroids = group_centroids / counts.unsqueeze(-1)

    return group_attr, group_centroids


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------

def _center_of_mass_and_spread(
    centroids: Tensor,  # [G, D]
    weights: Tensor,    # [G]
):
    weights = weights.abs()
    total = weights.sum()

    if total <= 0 or centroids.shape[0] == 0:
        D = centroids.shape[-1]
        return torch.zeros(D + 1, device=centroids.device)

    center = (centroids * weights.unsqueeze(-1)).sum(dim=0) / total

    d2 = ((centroids - center) ** 2).sum(dim=-1)
    spread = torch.sqrt((d2 * weights).sum() / total)

    return torch.cat([center, spread.unsqueeze(0)])  # [D+1]


def _grid_histogram(
    centroids: Tensor,  # [G, 2] (assumes normalized)
    weights: Tensor,    # [G]
    grid_size: int,
):
    weights = weights.abs()
    grid = torch.zeros(grid_size, grid_size, device=centroids.device)

    if centroids.shape[0] == 0 or weights.sum() <= 0:
        return grid

    xs = (centroids[:, 0] * grid_size).long().clamp(0, grid_size - 1)
    ys = (centroids[:, 1] * grid_size).long().clamp(0, grid_size - 1)

    for x, y, w in zip(xs, ys, weights):
        grid[y, x] += w

    if grid.sum() > 0:
        grid = grid / grid.sum()

    return grid.flatten()


def _weighted_distance(
    centroids: Tensor,
    weights: Tensor,
    target_centroid: Tensor,
):
    weights = weights.abs()
    total = weights.sum()

    if total <= 0 or centroids.shape[0] == 0:
        return torch.tensor(0.0, device=centroids.device)

    dists = torch.sqrt(((centroids - target_centroid) ** 2).sum(dim=-1))
    return (dists * weights).sum() / total


# ---------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------

def attribution_locality(
    attributions: tuple[Tensor, ...] | list[tuple[Tensor, ...]],
    feature_mask: tuple[Tensor, ...],
    centroids: tuple[Tensor, ...],
    grid_size: int = 5,
    targets: list[list[int]] | None = None,
    multi_target: bool = False,
    return_dict: bool = False,
):
    """
    Modality-agnostic attribution locality.

    Returns:
        center_of_mass : [B, M, D]
        spread         : [B, M]
        grid           : [B, M, G²]
        target_distance (optional)
    """
    is_list = isinstance(attributions, list)
    if not is_list:
        attributions = [attributions]

    outputs = []

    for attrs in attributions:
        bsz = attrs[0].shape[0]
        n_modalities = len(attrs)

        centers_out = []
        spreads_out = []
        grids_out = []
        target_out = [] if targets is not None else None

        for b in range(bsz):
            mod_centers = []
            mod_spreads = []
            mod_grids = []

            reduced_cache = []

            for m in range(n_modalities):
                attr = attrs[m][b].flatten().float()
                mask = feature_mask[m][b].flatten().long()
                cent = centroids[m][b]

                g_attr, g_cent = _reduce_groups(attr, mask, cent)
                reduced_cache.append((g_attr, g_cent))

                cs = _center_of_mass_and_spread(g_cent, g_attr)

                mod_centers.append(cs[:-1])
                mod_spreads.append(cs[-1])

                if g_cent.shape[-1] == 2:
                    grid = _grid_histogram(g_cent, g_attr, grid_size)
                else:
                    grid = torch.zeros(grid_size * grid_size, device=attr.device)

                mod_grids.append(grid)

            centers_out.append(torch.stack(mod_centers))
            spreads_out.append(torch.stack(mod_spreads))
            grids_out.append(torch.stack(mod_grids))

            # --- target distances ---
            if targets is not None:
                t_list = targets[b]
                t_mod = []

                for t in t_list:
                    per_mod = []
                    for (g_attr, g_cent) in reduced_cache:
                        if t < 0 or t >= g_cent.shape[0]:
                            continue
                        d = _weighted_distance(g_cent, g_attr, g_cent[t])
                        per_mod.append(d)
                    if len(per_mod) > 0:
                        t_mod.append(torch.stack(per_mod))

                if len(t_mod) > 0:
                    target_out.append(torch.stack(t_mod))
                else:
                    target_out.append(torch.zeros(0, n_modalities))

        result = {
            "center_of_mass": torch.stack(centers_out),   # [B, M, D]
            "spread": torch.stack(spreads_out),           # [B, M]
            "grid": torch.stack(grids_out),               # [B, M, G²]
        }

        if targets is not None:
            result["target_distance"] = target_out

        outputs.append(result)

    if not is_list:
        outputs = outputs[0]

    if return_dict:
        return {"score": outputs}

    return outputs