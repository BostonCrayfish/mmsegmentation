# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

# decide data directory by home name
# please remove these lines and directly set data_root for your training
# import os
# if '/home/feng' in os.getcwd():
#     pretrain_path = '/home/feng/work_mmseg/checkpoints/moco_r50_200ep_trans.pth'
# elif '/home/cwei' in os.getcwd():
#     pretrain_path = '/home/cwei/feng/work_mmseg/checkpoints/moco_r50_200ep_trans.pth'
# else:
#     raise ValueError('unknown data directory')

model = dict(
    type='EncoderDecoder',
    pretrained='/home/feng/work_mmseg/checkpoints/moco_r50_200ep_trans.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
