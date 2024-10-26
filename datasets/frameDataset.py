import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor

import numpy as np
from datasets.Wildtrack import Wildtrack
from datasets.Terrace import Terrace
from typing import Dict

class frameDataset(VisionDataset):
    def __init__(self, base=Wildtrack, train=True, transform=ToTensor(), transform_ori=ToTensor(), target_transform=ToTensor(), 
                 train_ratio=0.9, pseudo_mask_path='tmp'):
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        self.transform_ori = transform_ori

        self.train, self.base  = train, base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.img_reduce = base.img_reduce
        
        self.expand_width, self.expand_length = base.expand_width, base.expand_length
        self.bev_height, self.voxel_size = base.bev_height, base.voxel_size

        if train:
            self.frame_range = list(range(0, int(self.num_frame * train_ratio)))
        else:
            self.frame_range = list(range(int(self.num_frame * train_ratio), self.num_frame))

        self.img_fpaths:Dict[int, Dict] = self.base.get_image_fpaths(self.frame_range)
        self.pseudo_fpaths, self.frame_list = self.get_pseudo_fpaths(pseudo_mask_path, self.frame_range)

    def get_pseudo_fpaths(self, gt_path: str, frame_range: range):
        subset_list = sorted(os.listdir(gt_path))
        img_list = sorted(os.listdir(os.path.join(gt_path, subset_list[0])))
        
        pseudo_fpaths, frame_list = {}, []
        for subset in subset_list:
            subset_idx = int(subset[-1])-1
            pseudo_fpaths[subset_idx] = {}
            for imgname in img_list:
                frame_idx = int(imgname.split('/')[-1].split('.')[0])
                if frame_idx not in frame_range: continue

                pseudo_fpaths[subset_idx][frame_idx] = os.path.join(gt_path, subset, imgname)
                if subset_idx == 0: frame_list.append(frame_idx)
        
        return pseudo_fpaths, frame_list
    
    def load_img_file(self, fpath_dict: Dict[int, list], frame: int, 
                      transform=None, color_mode='RGB'):
        imgs = []
        for cam in range(self.num_cam):
            fpath = fpath_dict[cam][frame]
            img = Image.open(fpath).convert(color_mode)
            if transform is not None:
                img = transform(img)
            else:
                img = self.target_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

    def __getitem__(self, index):
        frame = self.frame_list[index]
        imgs_ori = self.load_img_file(self.img_fpaths, frame, 
                                        transform=self.transform_ori)
        imgs_norm = self.load_img_file(self.img_fpaths, frame,
                                        transform=self.transform)
    
        masks = self.load_img_file(self.pseudo_fpaths, frame, 
                                    color_mode='L')
        masks = masks.squeeze(1)

        return imgs_ori, imgs_norm, masks, frame

    def __len__(self):
        return len(self.frame_list)

class frameDatasetTerrace(frameDataset):
    def __init__(self, base=Terrace, train=True, transform=ToTensor(), transform_ori=ToTensor(), target_transform=ToTensor(),
                train_ratio=0.9, pseudo_mask_path='tmp'):
        VisionDataset.__init__(self, base.root, transform=transform, target_transform=target_transform)
        self.transform_ori = transform_ori

        self.train, self.base  = train, base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape

        self.expand_width, self.expand_length = base.expand_width, base.expand_length
        self.bev_height, self.voxel_size = base.bev_height, base.voxel_size

        # For Terrace
        if train:
            frame_range = []
            for fm in range(35, 2511, 25):
                frame_range.append(fm)
            for fm in range(25, 2501, 25):
                frame_range.append(fm)
            for fm in range(15, 2516, 25):
                frame_range.append(fm)
        else:
            frame_range = []
            for fm in range(2535, 5000, 25):
                frame_range.append(fm)
            for fm in range(2525, 5000, 25):
                frame_range.append(fm)
        
        self.frame_range = frame_range

        self.img_fpaths:Dict[int, Dict] = self.base.get_image_fpaths(self.frame_range)
        self.pseudo_fpaths, self.frame_list = self.get_pseudo_fpaths(pseudo_mask_path, self.frame_range)