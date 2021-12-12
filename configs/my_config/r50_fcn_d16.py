# Configs for SegCo pretraining
# fcn stride 16 (same as moco v2)

norm_cfg = dict(type='BN', requires_grad=True)
pretrain_path = '/home/cwei/feng/work_place/checkpoints/moco/moco_r50_800ep_trans.pth'

model = dict(
    type='EncoderDecoder',
    pretrained=pretrain_path,
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
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        num_convs=2,
        concat_input=False,
        contrast=True,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
