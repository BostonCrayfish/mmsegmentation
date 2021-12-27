# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/vit_base_patch16_224.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        # out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    neck=None,
    decode_head=dict(
        type='ASPPHead',
        in_channels=768,
        # in_index=3,
        channels=512,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=21,
        contrast=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable