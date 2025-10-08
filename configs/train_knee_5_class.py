_base_ = ['../../../../../_base_/datasets/sam_dataset_segmentation.py', '../../../../../_base_/models/mask2former_sam.py']
data_root = 'UltraSAM_DATA/UltraSAM/'

classes = ('abnormal', 'fat', 'femur', 'tendon')

model = dict(
    backbone=dict(
        init_cfg=dict(prefix="backbone.", checkpoint="weights/UltraSam.pth")
    ),
    panoptic_head=dict(
        num_things_classes=len(classes),
        loss_cls=dict(
            class_weight=[1.0] * len(classes) + [0.1]
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=len(classes),
    ),
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img=''),
        ann_file='',
    ),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img=''),
        ann_file='',
    ),
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img=''),
        ann_file='',
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator['ann_file'] = ''
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator['ann_file'] = ''
test_evaluator = orig_test_evaluator