import torch
import torch.nn.functional as F

def pearson_corr_coef(x: torch.Tensor, y: torch.Tensor, dim: int = 1, per_track: bool = False, reduce_dims = (-1,)) -> torch.Tensor:
    # Compute Pearson correlation between x and y along dim using cosine-similarity of zero-centered signals
    # Args:\n", " x, y: tensors with identical shape, e.g. [B, num_bins, num_tracks]
    # dim: dimension along which to correlate (default 1 == num_bins)
    # per_track: if True return per-track correlations with shape [B, num_tracks]
    # if False, average across reduce_dims (default last dim -> per-sample scalar [B])
    # reduce_dims: dims to reduce when per_track is False (passed to Tensor.mean)
    # Returns:
    # Tensor of correlations (per-track or averaged as requested)
    # center
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    # cosine similarity along dim -> shape [B, num_tracks]
    corr = F.cosine_similarity(x_centered, y_centered, dim=dim)
    if per_track:
        return corr
    return corr.mean(dim=reduce_dims)

def rankdata_torch(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Return ranks (1..N) along dim using an argsort-argsort approach.
    This is a simple, fast approximation of scipy.stats.rankdata(method='average')
    which works well for continuous model outputs. Ties are given sequential ranks (no averaging).
    """
    # argsort twice gives dense ranks starting at 0; add 1 to match 1..N
    order = x.argsort(dim=dim)
    ranks = torch.zeros_like(order, dtype=torch.float, device=x.device)
    ranks.scatter_(dim, order, torch.arange(1, order.shape[dim] + 1, device=x.device, dtype=torch.float))
    return ranks

def spearman_corr_coef(x: torch.Tensor, y: torch.Tensor, dim: int = 1, per_track: bool = False, reduce_dims = (-1,)) -> torch.Tensor:
    """Compute Spearman correlation by ranking x,y along dim then computing Pearson on ranks.
    Same return contract as pearson_corr_coef.
    """
    x_rank = rankdata_torch(x, dim=dim)
    y_rank = rankdata_torch(y, dim=dim)
    x_centered = x_rank - x_rank.mean(dim=dim, keepdim=True)
    y_centered = y_rank - y_rank.mean(dim=dim, keepdim=True)
    rho = F.cosine_similarity(x_centered, y_centered, dim=dim)
    if per_track:
        return rho
    return rho.mean(dim=reduce_dims)

# Example usage (expects preds and targets to be torch tensors shaped [B, num_bins, num_tracks]):
preds = torch.randn(4, 196608, 6) # example
targets = torch.randn_like(preds)
# per-track Pearson: shape [B, num_tracks]
pearson_pt = pearson_corr_coef(preds, targets, dim=1, per_track=True)
# mean-per-sample Pearson (averaged across tracks): shape [B]
pearson_mean = pearson_corr_coef(preds, targets, dim=1, per_track=False)
# same for Spearman:
spearman_pt = spearman_corr_coef(preds, targets, dim=1, per_track=True)
spearman_mean = spearman_corr_coef(preds, targets, dim=1, per_track=False)

print(pearson_pt.shape, pearson_mean.shape, spearman_pt.shape, spearman_mean.shape)