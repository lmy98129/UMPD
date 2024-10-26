import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

from sklearn.decomposition import PCA

from net.ZeroShotCLIP import ZeroShotCLIP
from config import Config

def dino_feature_extractaction(imgpath_list, dinov2_vitb14, transform):
    features = []
    imgs_tensor = []    
    for img_path in sorted(imgpath_list):
        img = Image.open(img_path).convert('RGB')
        imgs_tensor.append(transform(img)[:3].unsqueeze(0))
    
    with torch.no_grad():
        for img_tensor in imgs_tensor:
            features_dict = dinov2_vitb14.forward_features(img_tensor.cuda())
            features.append(features_dict['x_norm_patchtokens'])
    features = torch.stack(features)
    return features

def clip_coarse_mask(imgpath_list, zeroshot_clip: ZeroShotCLIP):
    score_maps = []
    with torch.no_grad():
        for img_path in sorted(imgpath_list):
            img: torch.Tensor = zeroshot_clip.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            score_map = zeroshot_clip(img.to(device)).float().cpu()
            score_map = torch.softmax(score_map.squeeze(0), dim=0)[0]
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min())
            score_maps.append(score_map) 

    return score_maps  

def clip_foreground_classifier(mask, score_maps, num_cameras):
    mask = torch.from_numpy(mask.reshape(num_cameras, patch_h, patch_w)).unsqueeze(1).clone()
    
    all_inter_sum = []
    for cur_mask in (mask, ~mask):
        cur_inter_sum = 0
        for score_map, cur_mask_item in zip(score_maps, cur_mask):
            cur_mask_tmp = F.interpolate(cur_mask_item.unsqueeze(1).float(), score_map.shape[-2:])
            intersection = score_map[((score_map > 0.6) & (cur_mask_tmp > 0)).squeeze(0).squeeze(0)]
            cur_inter_sum += intersection.sum().item()
        all_inter_sum.append(cur_inter_sum)
    
    return all_inter_sum[0] > all_inter_sum[1]

def princial_component_analysis(features: torch.Tensor, score_maps: torch.Tensor, 
                                pca_iteration: int, feature_dim: int):
    num_cameras = features.shape[0]
    pca_features = features.cpu().reshape(num_cameras * patch_h * patch_w, feature_dim).clone()

    foreground_mask_kept = None
    for iter_idx in range(pca_iteration):
        pca = PCA(n_components=1)
        pca.fit(pca_features)
        pca_results = pca.transform(pca_features)
        
        background_mask = pca_results[:, 0] < 2
        foreground_mask = ~background_mask
        
        if iter_idx == 0:
            if score_maps is not None:
                is_foreground = clip_foreground_classifier(foreground_mask, score_maps, num_cameras)
                if not is_foreground: foreground_mask = ~foreground_mask
            foreground_mask_kept = foreground_mask.copy()

        else:    
            if score_maps is not None:
                foreground_mask_tmp = foreground_mask_kept.copy()
                foreground_mask_tmp[foreground_mask_tmp] = foreground_mask
            
                is_foreground = clip_foreground_classifier(foreground_mask_tmp, score_maps, num_cameras)
                if not is_foreground: foreground_mask = ~foreground_mask
            
            foreground_mask_kept[foreground_mask_kept] = foreground_mask
        
        pca_features = pca_features[foreground_mask]

    foreground_masks = []
    for i in range(num_cameras):
        foreground_mask = foreground_mask_kept[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w)
        foreground_masks.append(foreground_mask)

    return foreground_masks

def main():
    cfg = Config()

    dinov2_vitb14 = torch.hub.load(cfg.dino_path, 'dinov2_vitb14', source='local')
    dinov2_vitb14.cuda()
    
    save_path = os.path.join('tmp', f'{cfg.dataset}Mask')
    if not os.path.exists(save_path): os.makedirs(save_path)

    data_path = os.path.join(cfg.data_path, 'Image_subsets')
    subset_list = sorted(os.listdir(data_path))
    
    imgname_list = sorted(os.listdir(os.path.join(data_path, subset_list[0])))

    global patch_h
    global patch_w

    # For CLIP to recognize small pedestrians
    clip_npx = 1500

    if cfg.dataset in ('Wildtrack', 'MultiviewX'):
        # 1080 // 14, 1920 // 14
        patch_h = 77
        patch_w = 137
    elif cfg.dataset == 'Terrace':
        # original, 360 // 14 is too low
        patch_h = 288
        patch_w = 360

    # more pca_iterations for too complex synthetic dataset
    pca_iteration = 3 if cfg.dataset == 'MultiviewX' else 2

    zeroshot_clip = ZeroShotCLIP(cfg.clip_path, clip_npx)
    transform = T.Compose([
        T.Resize((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for imgname in imgname_list:
        print(imgname)
        start_time = time.time()

        imgpath_list = [ os.path.join(data_path, subset, imgname) for subset in subset_list  ]
        score_maps = clip_coarse_mask(imgpath_list, zeroshot_clip) if cfg.use_clip else None
        print('clip mask generated')
        features = dino_feature_extractaction(imgpath_list, dinov2_vitb14, transform)
        print('dino feature extracted')
        foreground_masks = princial_component_analysis(features, score_maps, pca_iteration, cfg.dino_dim)
        print('pca iteration done')

        for subset, foreground_mask in zip(subset_list, foreground_masks):

            save_subset_path = os.path.join(save_path, subset)
            if not os.path.exists(save_subset_path): os.makedirs(save_subset_path)
            cur_foreground_mask = torch.from_numpy(~foreground_mask).float()
            plt.imsave(os.path.join(save_subset_path, imgname), 
                       cur_foreground_mask, cmap=matplotlib.colormaps['binary'])
        
        used_time = time.time() - start_time
        print(f'used {int(used_time // 60)} min {int(used_time % 60)} sec')
        
if __name__ == '__main__':
    main()