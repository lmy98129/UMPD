import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from utils.process import get_perspective_cameras

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures.volumes import VolumeLocator

from datasets import frameDataset

from net.resnet import resnet18
from typing import Dict, Tuple

class UMPD(nn.Module):
    def __init__(self, cfg: Config, dataset: frameDataset, patch_h, patch_w, 
                 worldgrid_shape, worldgrid_translation, device):
        super().__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.worldgrid_shape = worldgrid_shape
        self.worldgrid_translation = worldgrid_translation
        self.device = device

        self.patch_h, self.patch_w = patch_h, patch_w

        self.proj_params, self.volume_shape \
            = self.compute_projection(patch_h, patch_w)

        self.d_model = cfg.resnet_dim // 4

        self.feature_proj = nn.Sequential(
            nn.Conv2d(cfg.resnet_dim, self.d_model, kernel_size=1),
            nn.BatchNorm2d(self.d_model), nn.ReLU(),
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(self.d_model, self.d_model, kernel_size=1),
            nn.BatchNorm3d(self.d_model), nn.ReLU(),
            nn.Conv3d(self.d_model, self.d_model // 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(self.d_model // 2), nn.ReLU(),
            nn.Conv3d(self.d_model // 2, 4, kernel_size=1), 
            nn.Sigmoid(),
        )

        self.backbone = nn.Sequential(*list(resnet18(
                                replace_stride_with_dilation=[False, True, True]).children())[:-2])

    def forward(self, img_norm: torch.Tensor):
        B, N, __, H, W = img_norm.shape
        img_norm = img_norm.contiguous().view(B*N, 3, H, W).to(self.device)

        img_features = self.backbone(img_norm)
        img_features: torch.Tensor = self.feature_proj(img_features)
        img_features = img_features.contiguous()\
                                .view(B, N, self.d_model, self.patch_h, self.patch_w)

        volume_features = self.backproject(img_features)
        volume_features = (volume_features.softmax(dim=1) * volume_features).sum(dim=1)
        volume_features = volume_features.contiguous()\
                                    .view((B, -1, *self.volume_shape))
        
        volume_pred: torch.Tensor = self.conv3d(volume_features)

        volume_pred = volume_pred.split([3, 1], dim=1)
        return volume_pred
    
    def backproject(self, img_features: torch.Tensor):
        B, N, D, H, W = img_features.shape
        x, y, valid = self.proj_params['x'], self.proj_params['y'], self.proj_params['valid']
        P = valid.shape[-1]

        volume = torch.zeros((B, N, D, P), device=self.device)
        
        for j in range(N):
            volume[:, j, :, valid[j]] = img_features[:, j, :, y[j, valid[j]], x[j, valid[j]]]

        return volume
    
    def compute_projection(self, patch_h, patch_w) -> Tuple[Dict[str, torch.Tensor], list]:
        bev_width, bev_length = [ int(x * 0.025 * 100 // self.dataset.voxel_size) for x in self.worldgrid_shape]

        volume_shape = [self.dataset.bev_height, bev_length + self.dataset.expand_length, bev_width + self.dataset.expand_width]
        volume_locator = VolumeLocator(
            batch_size=1, grid_sizes=volume_shape, 
            device=self.device, voxel_size=self.dataset.voxel_size, 
            volume_translation=[*self.worldgrid_translation, 0]
        )
        
        xyz_grid = volume_locator.get_coord_grid().squeeze().view(1, -1, 3)
        cameras: PerspectiveCameras\
            = get_perspective_cameras(self.dataset, patch_h, patch_w).to(self.device)
        
        # transform xyz to the camera view coordinates
        xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz_grid)
        xyd = cameras.transform_points_screen(xyz_grid)
        
        x, y, d = xyd[:, :, 0].round().long(), xyd[:, :, 1].round().long(), xyz_cam[:, :, -1]

        valid = (x >= 0) & (y >= 0) & (x < patch_w) & (y < patch_h) & (d > 0)
        proj_params = { 'x': x, 'y': y, 'valid': valid }

        return proj_params, volume_shape