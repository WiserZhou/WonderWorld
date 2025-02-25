#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh import eval_sh
import torch.nn.functional as F
from utils.general import build_rotation, rotation2normal

def render(viewpoint_camera, pc: GaussianModel, opt, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, render_visible=False, exclude_sky=False):
    """
    Render the scene using 3D Gaussian splatting.
    
    Args:
        viewpoint_camera: Camera parameters for rendering viewpoint
        pc (GaussianModel): Point cloud model containing 3D Gaussians
        opt: Rendering options and parameters
        bg_color (torch.Tensor): Background color tensor (must be on GPU)
        scaling_modifier (float): Modifier for Gaussian scaling
        override_color (torch.Tensor, optional): Override colors of Gaussians
        render_visible (bool): Only render Gaussians marked as visible
        exclude_sky (bool): Exclude sky points from rendering
        
    Returns:
        dict: Dictionary containing:
            - render: Rendered image
            - viewspace_points: Screen space points for gradient computation
            - visibility_filter: Boolean mask of visible points
            - radii: Screen space radii of Gaussians
            - final_opacity: Final opacity values
            - depth: Depth values
            - median_depth: Median depth value
    """

    # Create zero tensor for computing gradients of 2D screen-space means
    # This tensor will accumulate gradients during backpropagation
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Calculate camera frustum parameters
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # Tangent of half horizontal field of view
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # Tangent of half vertical field of view

    # Configure rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width), 
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    # Create rasterizer with settings
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get Gaussian parameters from point cloud
    means3D = pc.get_xyz_all  # 3D positions
    means2D = screenspace_points  # 2D projected positions
    opacity = pc.get_opacity_all  # Opacity values

    # Handle 3D covariance computation
    scales = None
    rotations = None 
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        # Precompute 3D covariance if specified
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        # Otherwise use scales and rotations for covariance computation in rasterizer
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # Handle color computation
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            # Convert spherical harmonics to RGB in Python
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            colors_precomp = pc.color_activation(shs_view)
        else:
            # Use raw features for SH conversion in rasterizer
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    # Create visibility filter based on rendering options
    visibility_filter_all = ~pc.delete_mask_all  # Start with non-deleted points
    if render_visible:
        visibility_filter_all &= pc.visibility_filter_all  # Add visibility check
    if exclude_sky:
        visibility_filter_all &= ~pc.is_sky_filter  # Exclude sky points

    # Apply visibility filter to all parameters
    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Perform rasterization
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Return rendering results
    return {
        "render": rendered_image,  # Final rendered image
        "viewspace_points": screenspace_points,  # Screen space points for gradients
        "visibility_filter": radii > 0,  # Mask of points with non-zero radius
        "radii": radii,  # Screen space radii
        "final_opacity": final_opacity,  # Final opacity values
        "depth": depth,  # Depth values
        "median_depth": median_depth,  # Median depth
    }