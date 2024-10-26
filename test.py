import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config import Config
from datasets import *
from net import UMPD
from utils.eval import evaluation

import argparse

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save-path', default='outputs', 
                        help='Folder to save results at ./tmp')
    parser.add_argument('--weight-path', default='outputs', 
                        help='Model weight to evaluate')
    
    return parser.parse_args()

def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = frameDataset
    if cfg.dataset == 'Wildtrack':
        database = Wildtrack(cfg.data_path)
    elif cfg.dataset == 'MultiviewX':
        database = MultiviewX(cfg.data_path)
    elif cfg.dataset == 'Terrace':
        database = Terrace(cfg.data_path)
        dataset = frameDatasetTerrace
    else:
        raise Exception('must choose from [Wildtrack, MultiviewX, Terrace]')

    patch_h = database.img_shape[0] // database.img_reduce
    patch_w = database.img_shape[1] // database.img_reduce

    patch_h_resnet = database.img_shape[0] // (cfg.resnet_down * 2)
    patch_w_resnet = database.img_shape[1] // (cfg.resnet_down * 2)

    dino_transform = T.Compose([
        T.Resize((patch_h_resnet * cfg.resnet_down, patch_w_resnet * cfg.resnet_down)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dino_transform_ori = T.Compose([
        T.Resize((patch_h * database.img_reduce, patch_w * database.img_reduce)),
        T.ToTensor(),
    ])

    testdataset = dataset(database, train=False, transform=dino_transform, transform_ori=dino_transform_ori, 
                          pseudo_mask_path=cfg.pseudo_mask_path)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False,
                                                num_workers=4, pin_memory=True)
    
    worldgrid_translation = database.worldgrid_translation
    worldgrid_shape = database.worldgrid_shape
    if database.indexing == 'xy': 
        worldgrid_shape = worldgrid_shape[::-1]
        
    net = UMPD(cfg, testdataset, patch_h_resnet, patch_w_resnet,
                          worldgrid_shape, worldgrid_translation, device).to(device)        

    state_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()

    return net, database, testdataset, testloader, worldgrid_shape

if __name__ == '__main__':
    args, cfg = parse(), Config()
    save_path, weight_path = args.save_path, args.weight_path

    net, database, testdataset, testloader, worldgrid_shape = init()
    evaluation(net, database, testloader, save_path, device='cuda:0', 
               worldgrid_shape=worldgrid_shape, visualization=False, show_process=True)