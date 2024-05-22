"""Python bindings for 3D gaussian projection"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from gsplat._torch_impl import scale_rot_to_cov3d


def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"

    # cov3d = torch.eye(3, device=means3d.device, dtype=means3d.dtype).unsqueeze(0).expand(
    #     means3d.shape[0], -1, -1
    # )  # TODO COMPUTE ACTUAL COV

    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov3d_triu = pack_cov3d_triu(cov3d)

    return _ProjectGaussians.apply(
        means3d.contiguous(),
        cov3d_triu,
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )


def pack_cov3d_triu(cov3d):
    """
    Pack the triangular covariance matrices in the order expected by the CUDA code.
    """
    cov_xx = cov3d[..., 0, 0]
    cov_xy = cov3d[..., 0, 1]
    cov_xz = cov3d[..., 0, 2]
    cov_yy = cov3d[..., 1, 1]
    cov_yz = cov3d[..., 1, 2]
    cov_zz = cov3d[..., 2, 2]

    cov3d_triu = torch.stack([cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz], dim=-1)
    return cov3d_triu


class _ProjectGaussians(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        cov3d_triu: Float[Tensor, "*batch 6"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            cov3d_triu,  # Pass cov3d_triu instead of individual scale and quat tensors
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            viewmat,
            cov3d_triu,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, radii, conics, compensation, num_tiles_hit)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_depths,
        v_radii,
        v_conics,
        v_compensation,
        v_num_tiles_hit,
    ):
        (
            means3d,
            viewmat,
            cov3d_triu,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        v_cov2d, v_cov3d, v_mean3d = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            cov3d_triu,  # Pass cov3d_triu instead of individual scale and quat tensors
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_conics,
            v_compensation,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # ... (rest of the viewmat gradient computation) ...

        else:
            v_viewmat = None

        # Return a gradient for each input.
        return (
            v_mean3d,
            v_cov3d,  # Return v_cov2d instead of v_scale and v_quat
            v_viewmat,
            None,  # fx
            None,  # fy
            None,  # cx
            None,  # cy
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # clip_thresh
        )
