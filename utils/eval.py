import os

import cv2
import torch
from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm

from datasets import *
from net import UMPD

from utils.nms import nms
from matplotlib import pyplot as plt

def evaluation(net:UMPD, database:Wildtrack, testloader:DataLoader, save_path:str, device:str, 
               worldgrid_shape: list, visualization=True, show_process=False):
    base = database
    bev_width, bev_length = [ int(x * 0.025 * 100 // base.voxel_size) for x in worldgrid_shape]

    density_list, frame_list = [], []
    net.eval()

    if show_process: testloader = tqdm(testloader)
    for __, imgs_norm, ___, frame in testloader:
        imgs_norm: torch.Tensor = imgs_norm
        frame: torch.Tensor = frame

        with torch.no_grad():
            density_ori: torch.Tensor = net(imgs_norm.to(device))[1]

        density_list.append(density_ori)
        frame_list.append(frame)

    if database.indexing == 'xy':
        worldgrid_shape = database.worldgrid_shape[::-1]
    else:
        worldgrid_shape = database.worldgrid_shape
    
    cls_thres = 0.4
    all_res_list = []

    for frame, density in zip(frame_list, density_list):
        frame = torch.Tensor([frame])
        density:torch.Tensor = density.detach().cpu().squeeze().max(dim=0)[0]

        if database.indexing == 'xy' and database.__name__ != 'Terrace':
            density = density.flip(0).flip(1)
        
        if base.expand_length != 0 or base.expand_width != 0:
            density = density[base.expand_length//2-1:base.expand_length//2+bev_length-1, 
                                base.expand_width//2-1:base.expand_width//2+bev_width-1]

        if database.indexing == 'ij':
            density = density.permute(1, 0)

        density = density.cpu().numpy()

        if visualization: 
            plt.imshow(density)
            plt.savefig(os.path.join(save_path, f'{str(int(frame[0].item())).zfill(8)}.jpg'))

        density = cv2.GaussianBlur(density, (3, 3), database.sigma)

        map_grid_res = torch.from_numpy(density).squeeze()
        v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
        grid_ij = (map_grid_res > cls_thres).nonzero()
        if database.indexing == 'xy':
            grid_xy = grid_ij[:, [1, 0]]
        else:
            grid_xy = grid_ij
        all_res_list.append(torch.cat([torch.ones_like(v_s) * frame.float(), grid_xy.float() *
                                        (base.voxel_size / (0.025 * 100)), v_s], dim=1))
    
    all_res_list = torch.cat(all_res_list, dim=0)

    res_list = []
    for frame in np.unique(all_res_list[:, 0]):
        res = all_res_list[all_res_list[:, 0] == frame, :]
        positions, scores = res[:, 1:3], res[:, 3]
        ids, count = nms(positions, scores, 20, np.inf)
        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

    np.savetxt(os.path.join(save_path, 'test.txt'), res_list, '%d')