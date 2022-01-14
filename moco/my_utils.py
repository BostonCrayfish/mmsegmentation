import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from moco.moco import loader as moco_loader
import torchvision.transforms.functional as F
import math

def gen_mask(batch, image_size, num_blocks, kernel, ratio=1.25, proportion=(.5, .8), redundancy=2.):
    batch_r = int(batch * redundancy)

    # conv layer configs
    kmin = int(kernel / ratio)
    kmax = int(kernel * ratio)
    ka = torch.randint(kmin, kmax + 1, []) // 2 * 2 + 1
    kb = torch.randint(kmin, kmax + 1, []) // 2 * 2 + 1
    sweeper = nn.Conv2d(1, 1, (ka, kb), padding=(ka // 2, kb // 2), bias=False)
    nn.init.constant_(sweeper.weight, 1.)

    # mask generating
    mask = torch.rand(batch_r, image_size ** 2)
    mids = mask.sort()[0][:, num_blocks]
    mask = (mask < mids.unsqueeze(1)).float().view(batch_r, 1, image_size, image_size)
    with torch.no_grad():
        mask = (sweeper(mask) > 0.).float()

    # delete the masks that don't meet the
    # requirements of area proportion
    p_min, p_max = proportion
    p_masks = mask.mean(dim=(1, 2, 3))
    indices = torch.where(((p_masks < p_max) * (p_masks > p_min)) == 1)[0]
    if len(indices) < batch:
        return gen_mask(batch, image_size, num_blocks, kernel, ratio, proportion, redundancy)
        # raise ValueError('Qualified masks are not enough, please check the' +
        #                  ' parameters or increase the argument \'redundancy\'. ' +
        #                  'Expected {} masks, got {} only'.format(batch, len(indices)))
    else:
        return mask[indices][0: batch]

def inpoly2(vert, node, ftol=5.0e-14):
    vert = np.asarray(vert, dtype=np.float32)
    node = np.asarray(node, dtype=np.float32)

    STAT = np.full(
        vert.shape[0], False, dtype=np.bool_)
    BNDS = np.full(
        vert.shape[0], False, dtype=np.bool_)

    if node.size == 0: return STAT, BNDS

    indx = np.arange(0, node.shape[0] - 1)

    edge = np.zeros((
        node.shape[0], +2), dtype=np.int32)

    edge[:-1, 0] = indx + 0
    edge[:-1, 1] = indx + 1
    edge[-1, 0] = node.shape[0] - 1

    # ----------------------------------- prune points using bbox
    xdel = np.nanmax(node[:, 0]) - np.nanmin(node[:, 0])
    ydel = np.nanmax(node[:, 1]) - np.nanmin(node[:, 1])

    lbar = (xdel + ydel) / 2.0

    veps = (lbar * ftol)

    mask = np.logical_and.reduce((
        vert[:, 0] >= np.nanmin(node[:, 0]) - veps,
        vert[:, 1] >= np.nanmin(node[:, 1]) - veps,
        vert[:, 0] <= np.nanmax(node[:, 0]) + veps,
        vert[:, 1] <= np.nanmax(node[:, 1]) + veps)
    )

    vert = vert[mask]

    if vert.size == 0: return STAT, BNDS

    # ------------------ flip to ensure y-axis is the `long` axis
    xdel = np.amax(vert[:, 0]) - np.amin(vert[:, 0])
    ydel = np.amax(vert[:, 1]) - np.amin(vert[:, 1])

    lbar = (xdel + ydel) / 2.0

    if (xdel > ydel):
        vert = vert[:, (1, 0)]
        node = node[:, (1, 0)]

    # ----------------------------------- sort points via y-value
    swap = node[edge[:, 1], 1] < node[edge[:, 0], 1]
    temp = edge[swap]
    edge[swap, :] = temp[:, (1, 0)]

    # ----------------------------------- call crossing-no kernel
    stat, bnds = \
        _inpoly(vert, node, edge, ftol, lbar)

    # ----------------------------------- unpack array reindexing
    STAT[mask] = stat
    BNDS[mask] = bnds

    return STAT, BNDS

def _inpoly(vert, node, edge, ftol, lbar):
    """
    _INPOLY: the local pycode version of the crossing-number
    test. Loop over edges; do a binary-search for the first
    vertex that intersects with the edge y-range; crossing-
    number comparisons; break when the local y-range is met.

    """

    feps = ftol * (lbar ** +1)
    veps = ftol * (lbar ** +1)

    stat = np.full(
        vert.shape[0], False, dtype=np.bool_)
    bnds = np.full(
        vert.shape[0], False, dtype=np.bool_)

    # ----------------------------------- compute y-range overlap
    ivec = np.argsort(vert[:, 1], kind="quicksort")

    XONE = node[edge[:, 0], 0]
    XTWO = node[edge[:, 1], 0]
    YONE = node[edge[:, 0], 1]
    YTWO = node[edge[:, 1], 1]

    XMIN = np.minimum(XONE, XTWO)
    XMAX = np.maximum(XONE, XTWO)

    XMIN = XMIN - veps
    XMAX = XMAX + veps
    YMIN = YONE - veps
    YMAX = YTWO + veps

    YDEL = YTWO - YONE
    XDEL = XTWO - XONE

    EDEL = np.abs(XDEL) + YDEL

    ione = np.searchsorted(
        vert[:, 1], YMIN, "left", sorter=ivec)
    itwo = np.searchsorted(
        vert[:, 1], YMAX, "right", sorter=ivec)

    # ----------------------------------- loop over polygon edges
    for epos in range(edge.shape[0]):

        xone = XONE[epos]
        xtwo = XTWO[epos]
        yone = YONE[epos]
        ytwo = YTWO[epos]

        xmin = XMIN[epos]
        xmax = XMAX[epos]

        edel = EDEL[epos]

        xdel = XDEL[epos]
        ydel = YDEL[epos]

        # ------------------------------- calc. edge-intersection
        for jpos in range(ione[epos], itwo[epos]):

            jvrt = ivec[jpos]

            if bnds[jvrt]: continue

            xpos = vert[jvrt, 0]
            ypos = vert[jvrt, 1]

            if xpos >= xmin:
                if xpos <= xmax:
                    # ------------------- compute crossing number
                    mul1 = ydel * (xpos - xone)
                    mul2 = xdel * (ypos - yone)

                    if feps * edel >= abs(mul2 - mul1):
                        # ------------------- BNDS -- approx. on edge
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == yone) and (xpos == xone):
                        # ------------------- BNDS -- match about ONE
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == ytwo) and (xpos == xtwo):
                        # ------------------- BNDS -- match about TWO
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (mul1 <= mul2) and (ypos >= yone) \
                            and (ypos < ytwo):
                        # ------------------- advance crossing number
                        stat[jvrt] = not stat[jvrt]

            elif (ypos >= yone) and (ypos < ytwo):
                # ----------------------- advance crossing number
                stat[jvrt] = not stat[jvrt]

    return stat, bnds

def random_shape(image_size, k, r_min=None, epsilon=1e-10):
    if not r_min:
        r_min = image_size * 0.3
    r_img = image_size / 2. + .5

    theta = np.linspace(0, 2 * np.pi, k + 1)[0: -1]
    r_theta = np.min([
        np.abs(r_img / (np.sin(theta) + epsilon)),
        np.abs(r_img / (np.cos(theta) + epsilon))
    ], axis=0)
    r = np.random.rand(k) * (r_theta - r_min) + r_min
    locs = [(r * np.cos(theta)).astype(np.int32), (r * np.sin(theta)).astype(np.int32)]
    # locs = []
    # for i in range(k):
    #     theta = i * 2 * np.pi / k
    #     r_theta = min(np.abs(r_img / (np.sin(theta) + epsilon)),
    #                   np.abs(r_img / (np.cos(theta) + epsilon)))
    #     r = np.random.rand() * (r_theta - r_min) + r_min
    #     loc = [int(r * np.cos(theta)), int(r * np.sin(theta))]
    #     locs.append(loc)
    return np.asarray(locs).T + r_img

class Crop_with_mask(torch.nn.Module):
    def __init__(self, size=224, scale=(0.2, 1), ratio=(0.75, 1.33333333)):
        super().__init__()
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

        width, height = img.size
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

    def forward(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size), F.resized_crop(mask, i, j, h, w, self.size)

class Flip_with_mask(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask)
        return img, mask

def my_loader(path):
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert('RGB')
    image_two_crops = []
    for _ in range(2):
        img = aug(image)
        nodes = random_shape(224, 10)
        id_pos, _ = inpoly2(np.asarray(np.meshgrid(np.arange(224), np.arange(224))).reshape(2, -1).T, nodes)
        mask = torch.zeros(224 * 224)
        mask[id_pos == 1] = 1.
        image_two_crops.append(img * mask.view(224, 224))
    return image_two_crops

def loader_non_random(path):
    path_mask = path.replace('ImageNet', 'dino_mask').replace('.JPEG', '.png')

    trans_crop = Crop_with_mask(size=224, scale=(0.2, 1))
    trans_flip = Flip_with_mask()
    trans_img = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5)])
    trans_tensor = transforms.ToTensor()
    trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    image = Image.open(path).convert('RGB')
    mask = Image.open(path_mask)
    two_crop_img = []

    # two dino masks
    for _ in range(2):
        image, mask = trans_crop(image, image)
        image = trans_img(image)
        image, mask = trans_flip(image, mask)
        image, mask = trans_tensor(image), trans_tensor(mask)
        image = trans_norm(image) * mask
        two_crop_img.append(image)

    # one dino mask and one random rectangle
    # image_q, mask_q = trans_crop(image_PIL, mask_PIL)
    # image_q = trans_img(image_q)
    # image_q, mask_q = trans_flip(image_q, mask_q)
    # image_q, mask_q = trans_tensor(image_q), trans_tensor(mask_q)
    # mask_q = (mask_q[1] > 0.5).float()  # channel 1 of mask is representive
    # image_q = trans_norm(image_q) * mask_q
    # two_crop_img.append(image_q)
    #
    # trans_kf = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     trans_norm
    # ])
    # trans_kb = transforms.RandomErasing(p=1., scale=(0.5, 0.8), ratio=(0.8, 1.25), value=1.)
    # image_k = trans_kf(image_PIL)
    # mask_k = trans_kb(torch.zeros(1, 224, 224))[0]
    # image_k = image_k * mask_k
    # two_crop_img.append(image_k)

    return two_crop_img

if __name__ == '__main__':

    path = '/home/cwei/feng/data/ImageNet/train/n01440764/n01440764_8580.JPEG'
    print(loader_non_random(path))


    # mask_total = []
    # import time
    # start = time.time()
    # for i in range(8192):
    #     nodes = random_shape(224, 10)
    #     id_pos, _ = inpoly2(np.asarray(np.meshgrid(np.arange(224), np.arange(224))).reshape(2, -1).T, nodes)
    #     mask = torch.zeros(224 * 224, dtype=torch.bool)
    #     mask[id_pos == 1] = True
    #     mask_total.append(mask.view(1, 224, 224))
    #     if i % 128 == 127:
    #         end = time.time()
    #         print('128 masks done in {} seconds'.format(end - start))
    #         start = time.time()
    # mask_total = torch.cat(mask_total)
    # torch.save(mask_total, './masks.pth')



    # import matplotlib.pyplot as plt
    # plt.imshow(mask.view(224, 224).numpy())
    # plt.plot(nodes[:, 0], nodes[:, 1], 'b', linewidth=5)
    # plt.plot([nodes[-1, 0], nodes[0, 0]], [nodes[-1, 1], nodes[0, 1]], 'b', linewidth=5)
    # plt.show()
    # print(mask.mean())