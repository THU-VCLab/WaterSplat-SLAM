import einops
import lietorch
import torch


def as_SE3(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC

def as_SO3(X):
    if isinstance(X, lietorch.SO3):
        return X
    _, q, _ = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    R_WC = lietorch.SO3(q)
    return R_WC

def as_SE3_s(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC, s


def as_SE3_cuda(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC

def as_SO3_cuda(X):
    if isinstance(X, lietorch.SO3):
        return X
    _, q, _ = einops.rearrange(X.data.detach(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    R_WC = lietorch.SO3(q)
    return R_WC

def as_SE3_s_cuda(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC, s
