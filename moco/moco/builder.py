# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from mmseg.models import build_segmentor

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
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)

        self.encoder_q = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        self.encoder_k = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

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
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            self.queue[:, ptr:self.K] = keys[0:self.K - ptr].T
            self.queue[:, 0:ptr + batch_size - self.K] = keys[self.K - ptr:batch_size].T
        else:
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
        idx_shuffle = torch.randperm(batch_size_all).cuda()

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

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        # print(q.shape, mask_q.shape)
        # import time
        # time.sleep(10)

        # mask_q dim=(1, 2)
        q_pos = (torch.mul(q.permute(1, 0, 2, 3), mask_q).sum(dim=(2, 3))
                 / (mask_q.sum(dim=(1, 2)) + 1e-5)).T   # masked pooling
        q_pos = nn.functional.normalize(q_pos, dim=1)
        q_neg = (torch.mul(q.permute(1, 0, 2, 3), (1 - mask_q)).sum(dim=(2, 3))
                 / ((1 - mask_q).sum(dim=(1, 2)) + 1e-5)).T
        q_neg = nn.functional.normalize(q_neg, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            ###
            k_pos = (torch.mul(k.permute(1, 0, 2, 3), mask_k).sum(dim=(2, 3))
                     / (mask_k.sum(dim=(1, 2)) + 1e-5)).T
            k_pos = nn.functional.normalize(k_pos, dim=1)
            k_neg = (torch.mul(k.permute(1, 0, 2, 3), (1 - mask_k)).sum(dim=(2, 3))
                     / ((1 - mask_k).sum(dim=(1, 2)) + 1e-5)).T
            k_neg = nn.functional.normalize(k_neg, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)
        # negative logits: NxK (Nx(K+2)x196 in dense version)
        l_neg = torch.einsum('nc,ck->nk', [q_pos, self.queue.clone().detach()])

        # negative logits for backgrounds: Nx2
        l_neg_bg = torch.cat(
            [torch.einsum('nc,nc->n', [q_pos, q_neg]).unsqueeze(-1),
             torch.einsum('nc,nc->n', [q_pos, k_neg]).unsqueeze(-1)],
            dim=1)


        # try dense loss
        # logits: Nx(1+K) add line 172
        logits = torch.cat([l_pos, l_neg], dim=1)
        # logits for backgrounds: Nx3
        logits_bg = torch.cat([l_pos, l_neg_bg], dim=1)

        # apply temperature
        logits /= self.T
        logits_bg /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        # self._dequeue_and_enqueue(torch.cat([k_pos, k_neg], dim=0))
        self._dequeue_and_enqueue(k_pos)

        return logits, logits_bg, labels


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
