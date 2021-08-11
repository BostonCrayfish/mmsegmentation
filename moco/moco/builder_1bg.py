# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from mmseg.models import build_segmentor
import time
import torchvision.models as models
from mmseg.models.backbones.resnet import ResNet
from moco.moco import model

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, cfg, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # base_encoder = models.__dict__['resnet50']
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)
        # self.encoder_q = ResNet(50)
        # self.encoder_k = ResNet(50)
        self.encoder_q = model.Encoder_Decoder()
        self.encoder_k = model.Encoder_Decoder()

        # self.encoder_q = build_segmentor(
        #     cfg.model,
        #     train_cfg=cfg.get('train_cfg'),
        #     test_cfg=cfg.get('test_cfg'))
        # self.encoder_k = build_segmentor(
        #     cfg.model,
        #     train_cfg=cfg.get('train_cfg'),
        #     test_cfg=cfg.get('test_cfg'))

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        end = time.time()
        # idx_shuffle = torch.randperm(batch_size_all).cuda()
        idx_shuffle = torch.randperm(batch_size_all, device='cuda')
        print('shuffle time 1.2: ', time.time() - end)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, mask_q, mask_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        end = time.time()
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        print('line: 136, time: {}'.format(time.time() - end))
        end = time.time()
        # print(q.shape, mask_q.shape)
        # import time
        # time.sleep(10)

        # mask_q dim=(1, 2)
        # q_pos = (torch.mul(q.permute(1, 0, 2, 3), mask_q).sum(dim=(2, 3)) / mask_q.sum(dim=(1, 2))).T   # masked pooling
        # q_pos = nn.functional.normalize(q_pos, dim=1)
        # q_neg = (torch.mul(q.permute(1, 0, 2, 3), (1 - mask_q)).sum(dim=(2, 3)) / (1 - mask_q).sum(dim=(1, 2))).T
        # q_neg = nn.functional.normalize(q_neg, dim=1)
        q_pos = q.mean(dim=(2, 3))
        q_neg = q.mean(dim=(2, 3))
        # q_pos = q
        # q_neg = q

        print('line: 148, time: {}'.format(time.time() - end))
        end = time.time()

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            print('line: 154, time: {}'.format(time.time() - end))
            end = time.time()
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            print('line: 158, time: {}'.format(time.time() - end))
            end = time.time()
            k = self.encoder_k(im_k)  # keys: NxC
            print('line: 161, time: {}'.format(time.time() - end))
            end = time.time()
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            print('line: 165, time: {}'.format(time.time() - end))
            end = time.time()

            ###
            # k_pos = (torch.mul(k.permute(1, 0, 2, 3), mask_k).sum(dim=(2, 3)) / mask_k.sum(dim=(1, 2))).T
            # k_pos = nn.functional.normalize(k_pos, dim=1)
            # k_neg = (torch.mul(k.permute(1, 0, 2, 3), (1 - mask_k)).sum(dim=(2, 3)) / (1 - mask_k).sum(dim=(1, 2))).T
            # k_neg = nn.functional.normalize(k_neg, dim=1)
            k_pos = k.mean(dim=(2, 3))
            k_neg = k.mean(dim=(2, 3))
            # k_pos = k
            # k_neg = k

        print('line: 168, time: {}'.format(time.time() - end))
        end = time.time()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_fore_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)
        l_fore_neg = torch.einsum('nc,ck->nk', [q_pos, self.queue.clone().detach()])

        l_back_pos = torch.einsum('nc,nc->n', [q_neg, k_neg]).unsqueeze(-1)
        l_back_neg = torch.einsum('nc,ck->nk', [q_neg, self.queue.clone().detach()])

        l_seg = torch.cat(
            [torch.einsum('nc,nc->n', [q_pos, q_neg]).unsqueeze(-1),
             torch.einsum('nc,nc->n', [q_pos, k_neg]).unsqueeze(-1)],
            dim=1)

        logits_fore = torch.cat([l_fore_pos, l_fore_neg], dim=1)
        logits_back = torch.cat([l_back_pos, l_back_neg], dim=1)
        logits_seg = torch.cat([l_fore_pos, l_seg], dim=1)

        # apply temperature
        logits_fore /= self.T
        logits_back /= self.T
        logits_seg /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits_fore.shape[0], dtype=torch.long).cuda()
        print('line: 195, time: {}'.format(time.time() - end))
        end = time.time()
        # dequeue and enqueue
        self._dequeue_and_enqueue(torch.cat([k_pos, k_neg], dim=0))
        # self._dequeue_and_enqueue(k_pos)    # for moco_only baseline
        print('line: 200, time: {}'.format(time.time() - end))
        return logits_fore, logits_back, logits_seg, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
