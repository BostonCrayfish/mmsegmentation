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
        q = q.reshape(q.shape[0], q.shape[1], -1)    # queries: NxCx196
        mask_q = mask_q.reshape(-1)

        # mask negative points in q_dense
        q_dense = nn.functional.normalize(q, dim=1)
        idx_qpos = torch.where(mask_q == 1)[0]
        idx_qneg = torch.where(mask_q == 0)[0]
        q = q[:, :, idx_qpos]
        q_pos = nn.functional.normalize(q.mean(dim=2), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            k = k.reshape(k.shape[0], k.shape[1], -1)    # keys: NxCx196
            mask_k = mask_k.reshape(-1)
            idx_kpos = torch.where(mask_k == 1)[0]
            k_dense = nn.functional.normalize(k[:, :, idx_kpos], dim=1)
            k_pos = k[:, :, torch.where(mask_k == 1)[0]].mean(dim=2)
            k_pos = nn.functional.normalize(k_pos)

        # dense logits
        logits_dense = torch.einsum('ncx,ncy->nxy', [q_dense, k_dense])
        logits_dense = logits_dense.reshape(logits_dense.shape[0], -1)
        labels_dense = torch.einsum('x,y->xy', [mask_q, torch.ones_like(idx_kpos)]).reshape(-1)

        #################
        # postives = torch.where(labels_dense == 1.)[0]
        # negatives = torch.where(labels_dense == 0.)[0]
        # print(logits_dense[0, postives].mean().detach())
        # print(logits_dense[0, negatives].mean().detach())

        # dense logits q*q_pos
        # logits_dense_0 = torch.einsum('ncx,ncy->nxy', [q_dense[:, :, idx_qpos], q_dense[:, :, idx_qpos]])
        # logits_dense_0 = logits_dense_0.reshape(logits_dense_0.shape[0], -1)
        # len_q_pos = idx_qpos.shape[0]
        # idx_non_diag_q = torch.where(torch.eye(len_q_pos).view(-1) == 0)[0]
        # logits_dense_0 = logits_dense_0[:, idx_non_diag_q]
        # logits_dense_1 = torch.einsum('ncx,ncy->nxy', [q_dense[:, :, idx_qneg], q_dense[:, :, idx_qpos]])
        # logits_dense_1 = logits_dense_1.reshape(logits_dense_1.shape[0], -1)
        # logits_dense = torch.cat([logits_dense_0, logits_dense_1], dim=1)
        # labels_dense_0 = torch.ones(logits_dense_0.shape[1], dtype=torch.long).cuda()
        # labels_dense_1 = torch.zeros(logits_dense_1.shape[1], dtype=torch.long).cuda()
        # labels_dense = torch.cat([labels_dense_0, labels_dense_1])


        # moco logits
        l_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_pos, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)
        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).cuda()

        # apply temperature
        logits_moco /= self.T
        # logits_dense

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_pos)

        return logits_moco, logits_dense, labels_moco, labels_dense


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
