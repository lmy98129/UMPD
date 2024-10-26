import os
import numpy as np

import math
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from matplotlib import pyplot as plt
from matplotlib import gridspec
from io import TextIOWrapper

from datasets import frameDataset
from pytorch3d.renderer import PerspectiveCameras

def masked_images(silhouettes: torch.Tensor, flatten_imgs: torch.Tensor, thr: int=0):
    with torch.no_grad():
        target_images = []
        for silhouette, image in zip(silhouettes, flatten_imgs):
            masked_img = image.clone()
            masked_img: torch.Tensor = F.interpolate(masked_img.unsqueeze(0), silhouette.shape[-2:])
            masked_img = masked_img.squeeze(0)
            masked_img[silhouette.repeat(3, 1, 1) <= thr] = torch.tensor([0.2])
            target_images.append(masked_img.permute(1, 2, 0))

        target_images = torch.stack(target_images)

    return target_images

def visualize(frame_str, batch_idx, work_dir, indexing):
        # Function Closure
        def _visualize(rendered_images, target_images, rendered_silhouettes, target_silhouettes, density_map: torch.Tensor):
            plt.clf()
            plt.tight_layout()
            
            im_show_idx = int(np.random.randint(low=0, high=rendered_images.shape[0], size=(1,)))
            batch_show_idx = im_show_idx // (rendered_images.shape[0] // density_map.shape[0])

            fig = plt.figure()
            gs = gridspec.GridSpec(3, 2)
            
            ax = [
                fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, :])
            ]

            clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().squeeze().numpy()
            density_map = density_map[batch_show_idx].detach().cpu().squeeze().max(dim=0)[0]

            density_map = density_map.permute(1, 0) if indexing == 'ij' else density_map

            ax[0].imshow(clamp_and_detach(rendered_images[im_show_idx]))
            ax[1].imshow(clamp_and_detach(target_images[im_show_idx, ..., :3]))
            ax[2].imshow(clamp_and_detach(rendered_silhouettes[im_show_idx, ..., 0]))
            ax[3].imshow(clamp_and_detach(target_silhouettes[im_show_idx, ...]))
            ax[4].imshow(density_map)
            
            for ax_, title_ in zip(
                ax, 
                ("rendered image", "target image", "rendered silhouette", "target silhouette", "bev map")
            ):
                ax_.grid("off")
                ax_.axis("off")
                ax_.set_title(title_)
            fig.canvas.draw()
            fig.savefig(os.path.join(work_dir, f'frame_{frame_str+1}_iter_{str(batch_idx+1).zfill(3)}.png'), bbox_inches="tight")
            plt.close()

        return _visualize

def save_model(net: DDP, epoch: int, work_dir: str, log: TextIOWrapper):
    state_dict = net.module.state_dict()
    filename = f'epoch_{epoch+1}.pth'
    torch.save(state_dict, os.path.join(work_dir, filename))
    cur_log = '%s saved.' % filename
    print(cur_log)
    log.write(cur_log+'\n')
    log.flush()

def get_perspective_cameras(dataset: frameDataset, patch_h, patch_w):
    img_zoom_mat = np.diag(np.append(np.ones([2]) / (dataset.img_shape[0] // patch_h), [1]))

    RR, tt, f, p, img_size = [], [], [], [], []
    for i in range(dataset.num_cam):
        extrinsic_matrix = dataset.base.extrinsic_matrices[i].copy()
        intrinsic_matrix = dataset.base.intrinsic_matrices[i].copy()

        intrinsic_matrix_zoom = img_zoom_mat @ intrinsic_matrix
        fx, fy = intrinsic_matrix_zoom[0, 0], intrinsic_matrix_zoom[1, 1]
        px, py = intrinsic_matrix_zoom[0, -1], intrinsic_matrix_zoom[1, -1]

        R, t = extrinsic_matrix[:, :3], extrinsic_matrix[:, -1]
        
        # In PyTorch3D, we need to build the input first in order to define camera. Note that we consider batch size N = 1
        RR.append(torch.from_numpy(R).permute(1, 0)) # dim = (1, 3, 3)
        tt.append(torch.from_numpy(t)) # dim = (1, 3)
        f.append(torch.tensor((fx, fy), dtype=torch.float32)) # dim = (1, 2)
        p.append(torch.tensor((px, py), dtype=torch.float32)) # dim = (1, 2)
        img_size.append(torch.tensor((patch_h, patch_w), dtype=torch.float32)) # (height, width) of the image

    # Now, we can define the Perspective Camera model. 
    RR, tt, f, p, img_size = torch.stack(RR), torch.stack(tt), torch.stack(f), torch.stack(p), torch.stack(img_size)
    target_cameras = PerspectiveCameras(R=RR, T=tt, focal_length=-f, principal_point=p, image_size=img_size, in_ndc=False)

    return target_cameras