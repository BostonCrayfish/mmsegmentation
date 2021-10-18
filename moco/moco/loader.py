# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import torchvision.transforms.functional as F
import math


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class FixCrop(torch.nn.Module):
    def __init__(self, size=224, seed=0, seed_length=65536, scale=(0.2, 1), ratio=(0.75, 1.33333333)):
        super().__init__()
        torch.manual_seed(seed)
        self.seed_idx = 0
        self.seed_length = seed_length
        self.seed_generator = torch.randint(0, 8192, [self.seed_length])

        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale=(0.08, 1.), ratio=(0.75, 1.333)):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        random_seed = self.seed_generator[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % self.seed_length
        torch.manual_seed(random_seed)

        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size)

class RandomHorizontalFlip_FS(torch.nn.Module):
    def __init__(self, p=0.5, seed=0, seed_length=65536):
        super().__init__()
        self.p = p

        torch.manual_seed(seed)
        self.seed_idx = 0
        self.seed_length = seed_length
        self.seed_generator = torch.randint(0, 8192, [self.seed_length])

    def forward(self, img):
        random_seed = self.seed_generator[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % self.seed_length
        torch.manual_seed(random_seed)

        if torch.rand(1) < self.p:
            return F.hflip(img)
        return img