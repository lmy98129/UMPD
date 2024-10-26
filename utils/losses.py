import torch
from torch import nn

from config import Config
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    VolumeRenderer,
    PerspectiveCameras,
)
from pytorch3d.structures import Volumes
from types import FunctionType

import numpy as np
from datasets import frameDataset
from utils.process import *

class DifferentialRenderLoss(nn.Module):
    def __init__(self, cfg: Config, dataset: frameDataset, worldgrid_translation: list, patch_h: int, patch_w: int, device: str):
        super().__init__()
        self.cfg = cfg
        self.patch_h, self.patch_w = patch_h, patch_w

        # Our rendered scene is centered around (0,0,0) 
        # and is enclosed inside a bounding box

        # 1) Instantiate the raysampler.
        # Here, NDCMultinomialRaysampler generates a rectangular image
        # grid of rays whose coordinates follow the PyTorch3D
        # coordinate conventions.
        # Since we use a volume of size 128^3, we sample n_pts_per_ray=150,
        # which roughly corresponds to a one ray-point per voxel.
        # We further set the min_depth=0.1 since there is no surface within
        # 0.1 units of any camera plane.
        raysampler = NDCMultinomialRaysampler(
            image_width=patch_w,
            image_height=patch_h,
            n_pts_per_ray=200,
            min_depth=1,
            max_depth=4000,
        )

        # 2) Instantiate the raymarcher.
        # Here, we use the standard EmissionAbsorptionRaymarcher 
        # which marches along each ray in order to render
        # each ray into a single 3D color vector 
        # and an opacity scalar.
        raymarcher = EmissionAbsorptionRaymarcher()

        # Finally, instantiate the volumetric render
        # with the raysampler and raymarcher objects.
        self.renderer = VolumeRenderer(
            raysampler=raysampler, raymarcher=raymarcher,
        )

        self.volume_translation = [*worldgrid_translation, 0]

        self.dataset = dataset
        self.target_cameras = get_perspective_cameras(self.dataset, patch_h, patch_w)
        self.device = device

    def forward(self, densities: torch.Tensor, colors: torch.Tensor, 
                target_silhouettes: torch.Tensor, target_images: torch.Tensor, 
                view_idx: torch.Tensor, visfunc: FunctionType):
        batch_size = densities.shape[0]
        num_cameras = len(view_idx)

        cameras = PerspectiveCameras(
            focal_length=self.target_cameras.focal_length[view_idx]\
                .repeat(batch_size, 1).clone(),
            principal_point=self.target_cameras.principal_point[view_idx]\
                .repeat(batch_size, 1).clone(),
            R=self.target_cameras.R[view_idx].repeat(batch_size, 1, 1).clone(),
            T=self.target_cameras.T[view_idx].repeat(batch_size, 1).clone(),
            image_size=self.target_cameras.image_size[view_idx]\
                .repeat(batch_size, 1).clone(),
            in_ndc=False,
            device=self.device
        )
        
        volumes = Volumes(
            densities = densities.repeat(num_cameras, 1, 1, 1, 1),
            features = colors.repeat(num_cameras, 1, 1, 1, 1),
            voxel_size=self.dataset.voxel_size,
            volume_translation=self.volume_translation,
        )

        rendered_images, rendered_silhouettes =\
             self.renderer(cameras=cameras, volumes=volumes)[0].split([3, 1], dim=-1)

        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # target silhouettes.
        sil_err = self.huber(
            rendered_silhouettes[..., 0], target_silhouettes,
        ).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # target ground truth images.
        color_err = self.huber(
            rendered_images, target_images,
        ).abs().mean()

        if self.cfg.use_bev_err:
            bev_err = densities.squeeze().max(dim=0)[0].abs().mean()
        else:
            bev_err = torch.tensor([0.0]).to(self.device)

        if visfunc is not None: 
            visfunc(rendered_images, target_images, rendered_silhouettes, target_silhouettes, densities)

        return color_err, sil_err, bev_err
    
    @staticmethod
    def huber(x: torch.Tensor, y: torch.Tensor, scaling=0.1):
        assert x.shape == y.shape
        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
        return loss
