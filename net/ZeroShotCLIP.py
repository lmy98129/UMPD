import clip
import torch
import torch.nn as nn

from collections import OrderedDict
from clip.model import CLIP

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

class ZeroShotCLIP(nn.Module):
    def __init__(self, path='/path/to/CLIP/RN50.pt', npx=1500):
        super(ZeroShotCLIP, self).__init__()
        
        self.templates = ['a picture of {}']
        self.classnames = ['human', 'ground', 'sky']
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(path, device)

        self.preprocess = Compose([
            Resize(npx, interpolation=BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        self.visual = clip_model.visual
        self._init_visual(device)
        self._init_zeroshot_classifier(clip_model, device)

    def _init_visual(self, device):
        self.conv1 = nn.Conv2d(self.visual.attnpool.v_proj.in_features,
                               self.visual.attnpool.v_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        self.conv2 = nn.Conv2d(self.visual.attnpool.c_proj.in_features,
                               self.visual.attnpool.c_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        conv1_weight_shape = (*self.visual.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.visual.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.visual.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.visual.attnpool.c_proj.bias))

    @torch.no_grad()
    def _init_zeroshot_classifier(self, clip_model:CLIP, device):
        # refer to: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        zeroshot_weights = []
        for classname in self.classnames:
            texts = [template.format(classname) for template in self.templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(p=2)

            zeroshot_weights.append(class_embedding)

        # shape: [E, C]
        # where E is the dimension of an embedding and C is the number of classes.
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _stem(self, x):
        if hasattr(self.visual, 'relu'):
            for conv, bn in [(self.visual.conv1, self.visual.bn1),
                            (self.visual.conv2, self.visual.bn2),
                            (self.visual.conv3, self.visual.bn3)]:
                x = self.visual.relu(bn(conv(x)))
        else:
            x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
            x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
            x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))

        x = self.visual.avgpool(x)
        return x

    def encode_image(self, image):
        image = image.type(self.dtype)
        feature = self._stem(image)
        feature = self.visual.layer1(feature)
        feature = self.visual.layer2(feature)
        feature = self.visual.layer3(feature)
        feature = self.visual.layer4(feature)

        # removed attnpool
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        return feature

    def forward(self, images):
        # [B, E, h, w]
        features : torch.Tensor = self.encode_image(images)
        # Normalize
        features_norm = features / features.norm(p=2, dim=1, keepdim=True)
        # [B, w, h, E]
        features_t = features_norm.transpose(1, 3)
        # [B, w, h, C]
        score_map_t = features_t @ self.zeroshot_weights
        # [B, C, h, w]
        score_map = score_map_t.transpose(1, 3)

        return score_map