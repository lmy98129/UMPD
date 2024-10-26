import torch
import numpy as np
import random
import os
import argparse

from time import strftime, localtime
from config import Config

from datasets import *
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.trainer import Trainer

from net import UMPD
from torch.optim import Adam
from utils.process import save_model

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args()

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    cfg = Config()
    args = parse()
    local_rank = args.local_rank
    fix_seed(cfg.seed)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda:{}'.format(local_rank))

    time_log = strftime("%y%m%d-%H%M", localtime())
    work_dir = os.path.join('outputs', time_log, 'train')

    if local_rank == 0: 
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        cfg.print_conf()
        print('Training start')
        log_file = os.path.join(work_dir, time_log + '.log')
        log = open(log_file, 'w')
        cfg.write_conf(log)
    else:
        log = None

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

    traindataset = dataset(database, train=True, transform=dino_transform, transform_ori=dino_transform_ori, 
                           pseudo_mask_path=cfg.pseudo_mask_path)
    datasampler = DistributedSampler(traindataset)
    trainloader = DataLoader(traindataset, batch_size=cfg.one_gpu, shuffle=False, sampler=datasampler,
                                               num_workers=6, pin_memory=True)

    testdataset = dataset(database, train=False, transform=dino_transform, transform_ori=dino_transform_ori, 
                          pseudo_mask_path=cfg.pseudo_mask_path)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False,
                                                num_workers=4, pin_memory=True)

    worldgrid_translation = database.worldgrid_translation
    
    worldgrid_shape = database.worldgrid_shape
    if database.indexing == 'xy': 
        worldgrid_shape = worldgrid_shape[::-1]

    net = UMPD(cfg, traindataset, patch_h_resnet, patch_w_resnet,
                          worldgrid_shape, worldgrid_translation, device).to(device)

    net = DDP(net)

    optimizer = Adam(net.parameters(), lr=cfg.lr)

    trainer = Trainer(cfg, trainloader, testloader, optimizer, device, worldgrid_shape, worldgrid_translation, log, work_dir)

    for epoch in range(cfg.num_epochs):
        datasampler.set_epoch(epoch)

        if local_rank == 0:
            print('----------')
            print('Epoch %d begin' % ((epoch + 1)))

        trainer.train(epoch, net, local_rank)

        if local_rank == 0 and (epoch+1) % cfg.save_freq == 0 and (epoch+1) >= cfg.val_begin:
            save_model(net, epoch, work_dir, log)
            trainer.test(epoch, net, local_rank)
            

if __name__ == '__main__':
    main()