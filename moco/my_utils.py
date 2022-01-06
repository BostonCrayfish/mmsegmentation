import torch
from torch import nn


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


if __name__ == '__main__':
    mask = gen_mask(2, 14, 3, 9)