import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..utils.centernet.mapping import mapping as centernet_mapping

def get_config(client, name, owner=None, categories=[]):

    config = dict(
        task = 'detection', 
        categories = categories,

        save=True,
        save_dir='./output',
    
        display=True,
        display_it=50,

        model_path=None,

        train_dataset = {
            'name': 'detection',
            'kwargs': {
                'client': client,
                'name': name,
                'owner': owner,
                'labeled_only': True,
                'approved_only': False,
                'split': True,
                'train': True,
                'cache_location': './dataset',
                'fake_size': 1000,
                'transform':
                    A.Compose([
                        A.SmallestMaxSize(max_size=512),
                        A.RandomSizedCrop([200,300],256,256),
                        A.Flip(p=0.5),
                        A.Normalize(),
                        ToTensorV2(),
                    ], bbox_params=A.BboxParams('pascal_voc', label_fields=['category_ids'], min_visibility=0.35)),
                'mapping': centernet_mapping
            },
            'batch_size': 4,
            'workers': 4
        }, 

        val_dataset = {
            'name': 'detection',
            'kwargs': {
                'client': client,
                'name': name,
                'owner': owner,
                'labeled_only': True,
                'approved_only': False,
                'split': True,
                'train': False,
                'cache_location': './dataset',
                'transform':
                    A.Compose([
                        A.SmallestMaxSize(max_size=512),
                        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
                        A.Normalize(),
                        ToTensorV2(),
                    ], bbox_params=A.BboxParams('pascal_voc', label_fields=['category_ids'], min_visibility=0.35)),
                'mapping': centernet_mapping
            },
            'batch_size': 4,
            'workers': 4
        }, 

        model = {
            'name': 'unet', 
            'kwargs': {
                'encoder_name': 'resnet18',
                'encoder_weights': 'imagenet', 
                'classes': 3 + (len(categories) if len(categories) > 1 else 0),    
            },
            'init_output': True
        },

        loss_fn = {
            'name': 'centernet',
        },

        lr=5e-4,
        n_epochs=10
    )

    return config