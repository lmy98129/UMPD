import os
import time
import torch

import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from config import Config

from datasets import frameDataset, Wildtrack
from net import UMPD

from utils.losses import DifferentialRenderLoss
from io import TextIOWrapper

from utils.process import masked_images, visualize
from utils.eval import evaluation

class Trainer:
    def __init__(self, cfg:Config, trainloader:DataLoader, testloader:DataLoader, optimizer:optim.Adam,
                 device:str, worldgrid_shape:list, worldgrid_translation:list, log:TextIOWrapper, work_dir:str):
        self.trainloader = trainloader
        self.testloader = testloader

        dataset: frameDataset = trainloader.dataset
        self.dataset = dataset
        self.database: Wildtrack = self.dataset.base

        self.cfg = cfg
        self.device = device
        self.optimizer = optimizer

        self.patch_h = self.dataset.base.img_shape[0] // self.database.img_reduce
        self.patch_w = self.dataset.base.img_shape[1] // self.database.img_reduce

        self.loss = DifferentialRenderLoss(cfg, dataset, worldgrid_translation, 
                                           self.patch_h, self.patch_w, device)
        
        self.worldgrid_shape = worldgrid_shape

        self.best_loss = 100
        self.best_epoch = 1
        self.iter_cur = 0

        self.log = log
        self.work_dir = work_dir

        self.view_idx = list(range(dataset.num_cam))

    def train(self, epoch:int, net:UMPD, local_rank:int):
        if local_rank == 0:
            t1 = time.time() 
            t3 = time.time()

        net.train()
        total_loss_log, color_err_log, sil_err_log, bev_err_log, time_batch = 0, 0, 0, 0, 0
        epoch_loss = 0

        for batch_idx, (imgs_ori, imgs_norm, silhouettes, frame) in enumerate(self.trainloader):
            self.iter_cur += 1

            imgs_ori: torch.Tensor = imgs_ori
            silhouettes: torch.Tensor = silhouettes
            frame: torch.Tensor = frame
            
            B, N, C, H, W = imgs_ori.shape
            flatten_imgs_ori = imgs_ori.view(B*N, C, H, W)
            target_silhouettes = silhouettes.view(B*N, *silhouettes.shape[-2:])

            # BN, 3, H, W
            target_images = masked_images(target_silhouettes, flatten_imgs_ori)

            colors, densities = net(imgs_norm)

            total_loss, color_err, sil_err, bev_err = \
                    self.calculate_loss(densities, colors, target_silhouettes, target_images,
                                        batch_idx, epoch, use_vis_func=self.cfg.use_vis_func)

            self.optimizer.zero_grad()

            total_loss.backward()
            self.optimizer.step()

            total_loss_log += total_loss.item()
            color_err_log += color_err.item()
            sil_err_log += sil_err.item()
            bev_err_log += bev_err.item()
            epoch_loss += total_loss_log

            if (batch_idx+1) % self.cfg.log_freq == 0 and local_rank == 0:
                t4 = time.time()
                time_batch += (t4-t3)
                ETA_time = (len(self.trainloader) * self.cfg.num_epochs - self.iter_cur) * (time_batch/self.cfg.log_freq)
                self.write_iteration_log(batch_idx, epoch, time_batch, ETA_time, total_loss_log, 
                                         color_err_log, sil_err_log, bev_err_log)
                total_loss_log, color_err_log, sil_err_log, bev_err_log, time_batch = 0, 0, 0, 0, 0
                t3 = time.time()

        epoch_loss /= len(self.trainloader)
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch + 1

        if local_rank == 0:
            t2 = time.time()
            epoch_time = t2 - t1
            self.write_epoch_log(epoch, epoch_loss, epoch_time)

    def calculate_loss(self, densities, colors, 
                       target_silhouettes, target_images, 
                       batch_idx, epoch, use_vis_func=True):
        if use_vis_func and (batch_idx+1) % self.cfg.vis_freq == 0:
            vis_func = visualize(epoch, batch_idx, self.work_dir, self.dataset.base.indexing)
        else:
            vis_func = None
        
        color_err, sil_err, bev_err = self.loss(densities, colors, 
                                                target_silhouettes.to(self.device), target_images.to(self.device), 
                                                self.view_idx, vis_func)
        
        total_loss: torch.Tensor = self.cfg.color_lambda * color_err + \
                                            self.cfg.sil_lambda * sil_err + \
                                            self.cfg.bev_lambda * bev_err

        return total_loss, color_err, sil_err, bev_err 

    def write_iteration_log(self, batch_idx, epoch, time_batch, ETA_time, total_loss_log, 
                            color_err_log, sil_err_log, bev_err_log):
        m ,s = divmod(ETA_time, 60)
        h, m = divmod(m, 60)
        cur_log = '[Epoch %d/%d, Batch %d/%d]$ <Total loss: %.6f> color: %.6f, sil: %.6f, bev: %.6f, Time: %.3f, ETA: %d:%02d:%02d' %\
            (epoch + 1, self.cfg.num_epochs, batch_idx + 1, len(self.trainloader),
            total_loss_log/self.cfg.log_freq, color_err_log/self.cfg.log_freq, sil_err_log/self.cfg.log_freq, 
            bev_err_log/self.cfg.log_freq, time_batch/self.cfg.log_freq, h, m, s)
        
        print('\r'+cur_log, end='')
        self.log.write(cur_log+'\n')
        self.log.flush()

    def write_epoch_log(self, epoch, epoch_loss, epoch_time):
        cur_log = 'Epoch %d end, AvgLoss is %.6f, Time used %.1fsec.' % (epoch+1, epoch_loss, epoch_time)
        print('\r'+cur_log)
        self.log.write(cur_log+'\n')
        cur_log = 'Epoch %d has lowest loss: %.7f' % (self.best_epoch, self.best_loss)
        print('\r'+cur_log)
        self.log.write(cur_log+'\n')
        self.log.flush()

    def test(self, epoch: int, net:UMPD, local_rank:int):
        save_path = os.path.split(self.work_dir)[0]
        save_path = os.path.join(save_path, 'test', f'epoch_{str(epoch+1)}')

        if not os.path.exists(save_path): os.makedirs(save_path)

        evaluation(net, self.database, self.testloader, save_path, 
                   device=f'cuda:{local_rank}', worldgrid_shape=self.worldgrid_shape)