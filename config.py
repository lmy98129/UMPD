import os
from io import TextIOWrapper

class Config:
    def __init__(self):
        # seed
        self.seed = 1337
        # [Wildtrack, MultiviewX, Terrace]
        self.dataset = 'Terrace'

        # Paths
        # self.dino_path = '/path/to/dinov2'
        # self.clip_path = '/path/to/CLIP/RN50.pt'

        # self.pseudo_mask_path = os.path.join('tmp', f'{self.dataset}Mask')
        # self.data_path = os.path.join('/path/to/all/datasets/', self.dataset)

        # Example
        self.dino_path = '/home/prir1005/.cache/torch/hub/facebookresearch_dinov2_main'
        self.clip_path = '/home/prir1005/blean/torch-checkpoints/RN50.pt'

        subfixes = { 'Wildtrack': 'Mask', 'MultiviewX': 'MaskFull', 'Terrace': 'MaskHigh' }
        self.pseudo_mask_path = os.path.join('tmp', f'{self.dataset}{subfixes[self.dataset]}')
        
        self.data_path = os.path.join('/home/prir1005/pubdata/', self.dataset)
        
        # DINOv2, CLIP, ResNet18
        self.dino_dim = 768
        self.use_clip = True
        self.resnet_down = 8
        self.resnet_dim = 512

        # num_gpus * one_gpu, cameras, resnet_dim, patch_h, path_w
        self.one_gpu = 1
        self.num_epochs = 60
        self.lr =1e-2

        self.log_freq = 1
        self.vis_freq = 10
        
        # For larger test set of Terrace, val_begin = 40 to save time
        self.save_freq = 1
        self.val_begin = 1

        self.use_vis_func = True
        self.use_bev_err = True
        
        self.color_lambda = 1
        self.sil_lambda = 1
        self.bev_lambda = 1

    def print_conf(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))
    
    def write_conf(self, log: TextIOWrapper):
        log.write('\n'.join(['%s:%s' % item for item in self.__dict__.items()])+'\n')
        log.flush()