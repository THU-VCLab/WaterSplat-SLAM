"""Pure PyTorch implementations of various functions"""
# change from https://github.com/booqo/lowlight-underwater-for-nerf-studio/blob/master/lowlight_underwater/cudalight/_torch_impl.py
# and https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_torch_impl.py
import struct

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Tuple, Optional
from jaxtyping import Float, Int
from ..utils.utils import bin_and_sort_gaussians, compute_cumulative_intersects
from typing_extensions import Literal, assert_never

def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_matrix(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
) -> Tensor:
    """Convert quaternion and scale to a 3x3 matrix (R * S)."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)
    M = R * scales[..., None, :]  # (..., 3, 3)
    return M


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _fisheye_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of fisheye projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    x, y, z = torch.unbind(means, dim=-1)  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]

    eps = 0.0000001
    xy_len = (x**2 + y**2) ** 0.5 + eps
    theta = torch.atan2(xy_len, z + eps)
    means2d = torch.stack(
        [
            x * fx * theta / xy_len + cx,
            y * fy * theta / xy_len + cy,
        ],
        dim=-1,
    )

    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = torch.atan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / (x2y2)
    J = torch.stack(
        [
            fx * (x2 * a + y2 * b),
            fx * xy * (a - b),
            -fx * x * x2y2z2_inv,
            fy * xy * (a - b),
            fy * (y2 * a + x2 * b),
            -fy * y * x2y2z2_inv,
        ],
        dim=-1,
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _ortho_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of orthographic projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]

    O = torch.zeros((C, 1), device=means.device, dtype=means.dtype)
    J = torch.stack([fx, O, O, O, fy, O], dim=-1).reshape(C, 1, 2, 3).repeat(1, N, 1, 1)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = (
        means[..., :2] * Ks[:, None, [0, 1], [0, 1]] + Ks[:, None, [0, 1], [2, 2]]
    )  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def _fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    tmp = torch.sqrt(torch.clamp(b**2 - det, min=0.01))
    v1 = b + tmp  # (...,)
    r1 = 3.33 * torch.sqrt(v1)
    radius_x = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 0, 0]), r1))
    radius_y = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 1, 1]), r1))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # (..., 2)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= (radii > 0.0).all(dim=-1)

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id, 0] <= 0.0 or radii[cam_id, gauss_id, 1] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids[curr_idx] = (
                    (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                )
                flatten_ids[curr_idx] = index  # flattened index
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, flatten_ids


@torch.no_grad()
def _isect_offset_encode(
    isect_ids: Tensor, C: int, tile_width: int, tile_height: int
) -> Tensor:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_offset_encode()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros(
        (C, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    colors: Tensor,  # [C, N, channels]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor]:
    """Alpah compositing of 2D Gaussians in Pure Pytorch.

    This function performs alpha compositing for Gaussians based on the pair of indices
    {gaussian_ids, pixel_ids, camera_ids}, which annotates the intersection between all
    pixels and Gaussians. These intersections can be accquired from
    `gsplat.rasterize_to_indices_in_range`.

    .. note::

        This function exposes the alpha compositing process into pure Pytorch.
        So it relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        conics: Inverse of the 2D Gaussian covariance, Only upper triangle values. [C, N, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is
            enabled, Gaussian in each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated colors. [C, image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [C, image_height, image_width, 1]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gaussian_ids]  # [M, 2]
    c = conics[camera_ids, gaussian_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(
        opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    )

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )
    renders = accumulate_along_rays(
        weights,
        colors[camera_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, channels)
    alphas = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, 1)

    return renders, alphas


def _rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_colors = torch.zeros(
        (C, image_height, image_width, colors.shape[-1]), device=device
    )
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step = accumulate(
            means2d,
            conics,
            opacities,
            colors,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
        )
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]

    render_alphas = render_alphas
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[:, None, None, :] * (
            1.0 - render_alphas
        )

    return render_colors, render_alphas

def compute_sh_color(
    viewdirs: Float[Tensor, "*batch 3"], sh_coeffs: Float[Tensor, "*batch D C"]
):
    """
    :param viewdirs (*, C)
    :param sh_coeffs (*, D, C) sh coefficients for each color channel
    return colors (*, C)
    """
    *dims, dim_sh, C = sh_coeffs.shape
    bases = eval_sh_bases(dim_sh, viewdirs)  # (*, dim_sh)
    return (bases[..., None] * sh_coeffs).sum(dim=-2)


"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (
                        xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
    return result


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y ** 2 + z ** 2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x ** 2 + z ** 2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x ** 2 + y ** 2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    # assert(False)
    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))


def scale_rot_to_cov3d(scale: Tensor, glob_scale: float, quat: Tensor) -> Tensor:
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    R = normalized_quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    # TODO: save upper right because symmetric
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def project_cov3d_ewa(
    mean3d: Tensor,
    cov3d: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    tan_fovx: float,
    tan_fovy: float,
) -> Tuple[Tensor, Tensor]:
    assert mean3d.shape[-1] == 3, mean3d.shape
    assert cov3d.shape[-2:] == (3, 3), cov3d.shape
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape
    W = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    t = torch.einsum("...ij,...j->...i", W, mean3d) + p  # (..., 3)

    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz ** 2  # (...,)

    lim_x = 1.3 * torch.tensor([tan_fovx], device=mean3d.device)
    lim_y = 1.3 * torch.tensor([tan_fovy], device=mean3d.device)
    x_clamp = t[..., 2] * torch.clamp(t[..., 0] * rz, min=-lim_x, max=lim_x)
    y_clamp = t[..., 2] * torch.clamp(t[..., 1] * rz, min=-lim_y, max=lim_y)
    t = torch.stack([x_clamp, y_clamp, t[..., 2]], dim=-1)

    O = torch.zeros_like(rz)
    J = torch.stack(
        [fx * rz, O, -fx * t[..., 0] * rz2, O, fy * rz, -fy * t[..., 1] * rz2],
        dim=-1,
    ).reshape(*rz.shape, 2, 3)
    T = torch.matmul(J, W)  # (..., 2, 3)
    cov2d = torch.einsum("...ij,...jk,...kl->...il", T, cov3d, T.transpose(-1, -2))
    # add a little blur along axes and (TODO save upper triangular elements)
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + 0.3
    det_blur = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0))
    return cov2d[..., :2, :2], compensation


def compute_compensation(cov2d_mat: Tensor):
    """
    params: cov2d matrix (*, 2, 2)
    returns: compensation factor as calculated in project_cov3d_ewa
    """
    det_denom = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    det_nomin = (cov2d_mat[..., 0, 0] - 0.3) * (cov2d_mat[..., 1, 1] - 0.3) - cov2d_mat[
        ..., 0, 1
    ] ** 2
    return torch.sqrt(torch.clamp(det_nomin / det_denom, min=0))


def compute_cov2d_bounds(cov2d_mat: Tensor):
    """
    param: cov2d matrix (*, 2, 2)
    returns: conic parameters (*, 3)
    """
    det_all = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    valid = det_all != 0
    # det = torch.clamp(det, min=eps)
    det = det_all[valid]
    cov2d = cov2d_mat[valid]
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # (..., 3)
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    radius_all = torch.zeros(*cov2d_mat.shape[:-2], device=cov2d_mat.device)
    conic_all = torch.zeros(*cov2d_mat.shape[:-2], 3, device=cov2d_mat.device)
    radius_all[valid] = radius
    conic_all[valid] = conic
    return conic_all, radius_all, valid


def project_pix(fxfy, p_view, center, eps=1e-6):
    fx, fy = fxfy
    cx, cy = center

    rw = 1.0 / (p_view[..., 2] + 1e-6)
    p_proj = (p_view[..., 0] * rw, p_view[..., 1] * rw)
    u, v = (p_proj[0] * fx + cx, p_proj[1] * fy + cy)
    return torch.stack([u, v], dim=-1)


def clip_near_plane(p, viewmat, clip_thresh=0.01):
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    p_view = torch.einsum("ij,nj->ni", R, p) + T[None]
    return p_view, p_view[..., 2] < clip_thresh


def get_tile_bbox(pix_center, pix_radius, tile_bounds, block_width):
    tile_size = torch.tensor(
        [block_width, block_width], dtype=torch.float32, device=pix_center.device
    )
    tile_center = pix_center / tile_size
    tile_radius = pix_radius[..., None] / tile_size

    top_left = (tile_center - tile_radius).to(torch.int32)
    bottom_right = (tile_center + tile_radius).to(torch.int32) + 1
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    return tile_min, tile_max


def project_gaussians_forward(
    means3d,
    scales,
    glob_scale,
    quats,
    viewmat,
    intrins,
    img_size,
    block_width,
    clip_thresh=0.01,
):
    tile_bounds = (
        (img_size[0] + block_width - 1) // block_width,
        (img_size[1] + block_width - 1) // block_width,
        1,
    )
    fx, fy, cx, cy = intrins
    tan_fovx = 0.5 * img_size[0] / fx
    tan_fovy = 0.5 * img_size[1] / fy
    p_view, is_close = clip_near_plane(means3d, viewmat, clip_thresh)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov2d, compensation = project_cov3d_ewa(
        means3d, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    )
    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    xys = project_pix((fx, fy), p_view, (cx, cy))
    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    compensation = torch.where(~mask, 0, compensation)
    num_tiles_hit = torch.where(~mask, 0, num_tiles_hit)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    i, j = torch.triu_indices(2, 2)
    cov2d_triu = cov2d[..., i, j]
    # return (
    #     cov3d_triu,
    #     cov2d_triu,
    #     xys,
    #     depths,
    #     radii,
    #     conic,
    #     compensation,
    #     num_tiles_hit,
    #     mask,
    # )
    return (xys, depths, radii, conic, compensation, num_tiles_hit, cov3d)


def map_gaussian_to_intersects(
    num_points, xys, depths, radii, cum_tiles_hit, tile_bounds, block_width
):
    num_intersects = cum_tiles_hit[-1]
    isect_ids = torch.zeros(num_intersects, dtype=torch.int64, device=xys.device)
    gaussian_ids = torch.zeros(num_intersects, dtype=torch.int32, device=xys.device)

    for idx in range(num_points):
        if radii[idx] <= 0:
            break

        tile_min, tile_max = get_tile_bbox(
            xys[idx], radii[idx], tile_bounds, block_width
        )

        cur_idx = 0 if idx == 0 else cum_tiles_hit[idx - 1].item()

        # Get raw byte representation of the float value at the given index
        raw_bytes = struct.pack("f", depths[idx])

        # Interpret those bytes as an int32_t
        depth_id_n = struct.unpack("i", raw_bytes)[0]

        for i in range(tile_min[1], tile_max[1]):
            for j in range(tile_min[0], tile_max[0]):
                tile_id = i * tile_bounds[0] + j
                isect_ids[cur_idx] = (tile_id << 32) | depth_id_n
                gaussian_ids[cur_idx] = idx
                cur_idx += 1

    return isect_ids, gaussian_ids


def get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds):
    tile_bins = torch.zeros(
        (tile_bounds[0] * tile_bounds[1], 2),
        dtype=torch.int32,
        device=isect_ids_sorted.device,
    )

    for idx in range(num_intersects):
        cur_tile_idx = isect_ids_sorted[idx] >> 32

        if idx == 0:
            tile_bins[cur_tile_idx, 0] = 0
            continue

        if idx == num_intersects - 1:
            tile_bins[cur_tile_idx, 1] = num_intersects
            break

        prev_tile_idx = isect_ids_sorted[idx - 1] >> 32

        if cur_tile_idx != prev_tile_idx:
            tile_bins[prev_tile_idx, 1] = idx
            tile_bins[cur_tile_idx, 0] = idx

    return tile_bins


def rasterize_forward(
    tile_bounds,
    block,
    img_size,
    gaussian_ids_sorted,
    tile_bins,
    xys,
    conics,
    colors,
    opacities,
    background,
):
    channels = colors.shape[1]
    out_img = torch.zeros(
        (img_size[1], img_size[0], channels), dtype=torch.float32, device=xys.device
    )
    final_Ts = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.float32, device=xys.device
    )
    final_idx = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.int32, device=xys.device
    )
    for i in range(img_size[1]):
        for j in range(img_size[0]):
            tile_id = (i // block[0]) * tile_bounds[0] + (j // block[1])
            tile_bin_start = tile_bins[tile_id, 0]
            tile_bin_end = tile_bins[tile_id, 1]
            T = 1.0

            for idx in range(tile_bin_start, tile_bin_end):
                gaussian_id = gaussian_ids_sorted[idx]
                conic = conics[gaussian_id]
                center = xys[gaussian_id]
                delta = center - torch.tensor(
                    [j, i], dtype=torch.float32, device=xys.device
                )

                sigma = (
                    0.5
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                )

                if sigma < 0:
                    continue

                opac = opacities[gaussian_id]
                alpha = min(0.999, opac * torch.exp(-sigma))

                if alpha < 1 / 255:
                    continue

                next_T = T * (1 - alpha)

                if next_T <= 1e-4:
                    idx -= 1
                    break

                vis = alpha * T

                out_img[i, j] += vis * colors[gaussian_id]
                T = next_T

            final_Ts[i, j] = T
            final_idx[i, j] = idx
            out_img[i, j] += T * background

    return out_img, final_Ts, final_idx


def rasterize_gaussians_forward(
    xys: Float[Tensor, "*batch 2"],  # 高斯分布的二维坐标 (x, y)
    depths: Float[Tensor, "*batch 1"],  # 高斯分布的深度（与摄像机的相对位置）
    radii: Float[Tensor, "*batch 1"],  # 高斯分布的半径
    conics: Float[Tensor, "*batch 3"],  # 高斯分布的共轭矩阵（协方差矩阵的逆，通常用于描述高斯分布的形状）
    num_tiles_hit: Int[Tensor, "*batch 1"],  # 每个高斯分布影响的瓦片数
    colors: Float[Tensor, "*batch channels"],  # 高斯分布的颜色
    opacity: Float[Tensor, "*batch 1"],  # 高斯分布的透明度
    img_height: int,  # 渲染图像的高度
    img_width: int,  # 渲染图像的宽度
    block_width: int,  # 块的宽度，表示每个块包含的像素数量（2到16之间的整数）
    background: Optional[Float[Tensor, "channels"]] = None,  # 背景颜色（可选）
    return_alpha: Optional[bool] = False,  # 是否返回 alpha 通道（透明度）
) -> Tensor:
    """
       通过对每个瓦片的高斯分布交集进行排序和分组，光栅化 2D 高斯分布，并返回使用 alpha 合成的 N 维输出图像。

       参数:
           xys (Tensor): 高斯分布的 xy 坐标。
           depths (Tensor): 高斯分布的深度。
           radii (Tensor): 高斯分布的半径。
           conics (Tensor): 高斯分布的共轭矩阵（协方差的逆）。
           num_tiles_hit (Tensor): 每个高斯分布影响的瓦片数。
           colors (Tensor): 高斯分布的颜色。
           opacity (Tensor): 高斯分布的透明度。
           img_height (int): 渲染图像的高度。
           img_width (int): 渲染图像的宽度。
           block_width (int): 每个块的宽度，必须在 2 和 16 之间。
           background (Tensor): 背景颜色（可选）。
           return_alpha (bool): 是否返回 alpha 通道（透明度）。

       返回:
           一个 Tensor：
           - out_img (Tensor): 渲染后的图像。
           - out_alpha (Optional[Tensor]): 渲染后的 alpha 通道（可选）。
    """
    # 确保块宽度在2到16之间
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"

    # 如果颜色是 uint8 类型，则将其转换为浮动的 [0, 1] 范围
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255

    # 如果指定了背景颜色，则检查背景颜色的形状是否与颜色的通道数匹配
    if background is not None:
        assert background.shape[0] == colors.shape[
            -1], f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        # 如果没有指定背景，则使用全白背景
        background = torch.ones(colors.shape[-1], dtype=torch.float32, device=colors.device)

    # 验证 `xys` 和 `colors` 的维度
    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    # 计算图像大小和块的大小
    num_points = xys.size(0)  # 高斯分布的数量
    tile_bounds = (
        (img_width + block_width - 1) // block_width,  # 计算横向瓦片数
        (img_height + block_width - 1) // block_width,  # 计算纵向瓦片数
        1,
    )
    block = (block_width, block_width, 1)  # 每个瓦片的大小
    img_size = (img_width, img_height, 1)  # 图像的大小

    # 计算交集数和每个高斯影响的累积瓦片数
    num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

    # 如果没有交集，则创建背景图像
    if num_intersects < 1:
        out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
        )
        gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
        tile_bins = torch.zeros(0, 2, device=xys.device)
        final_Ts = torch.zeros(img_height, img_width, device=xys.device)
        final_idx = torch.zeros(img_height, img_width, device=xys.device)
    else:
        # 如果有交集，执行高斯分布的排序和分组
        isect_ids_unsorted, gaussian_ids_unsorted, isect_ids_sorted, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
            num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds, block_width
        )

        # 根据颜色的通道数选择光栅化函数
        if colors.shape[-1] == 3:
            rasterize_fn = rasterize_forward
        else:
            rasterize_fn = rasterize_forward

        # 调用光栅化函数生成图像
        out_img, final_Ts, final_idx = rasterize_fn(
            tile_bounds, block, img_size, gaussian_ids_sorted, tile_bins, xys, conics, colors, opacity, background
        )

    # 如果需要，返回 alpha 通道（透明度）
    if return_alpha:
        out_alpha = 1 - final_Ts  # alpha 通道是 1 - 透明度
        return out_img, out_alpha
    else:
        return out_img


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c
