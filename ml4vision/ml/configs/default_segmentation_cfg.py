import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_config(client, name, owner=None, categories=[]):

    n_classes = len(categories)

    config = dict(
        task = 'segmentation',
        categories = categories,

        save=True,
        save_dir='./output',
    
        display=True,
        display_it=50,

        model_path=None,

        train_dataset = {
            'name': 'segmentation',
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
                'ignore_zero': True if n_classes > 1 else False,
                'transform':
                    A.Compose([
                        A.SmallestMaxSize(max_size=512),
                        A.RandomSizedCrop([200,300],256,256),
                        A.Flip(p=0.5),
                        A.Normalize(),
                        ToTensorV2(),
                    ]),
            },
            'batch_size': 4,
            'workers': 4
        }, 

        val_dataset = {
            'name': 'segmentation',
            'kwargs': {
                'client': client,
                'name': name,
                'owner': owner,
                'labeled_only': True,
                'approved_only': False,
                'split': True,
                'train': False,
                'cache_location': './dataset',
                'ignore_zero': True if n_classes > 1 else False,
                'transform':
                    A.Compose([
                        A.SmallestMaxSize(max_size=512),
                        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
                        A.Normalize(),
                        ToTensorV2(),
                    ]),
            },
            'batch_size': 4,
            'workers': 4
        }, 

        model = {
            'name': 'unet', 
            'kwargs': {
                'encoder_name': 'resnet18',
                'encoder_weights': 'imagenet', 
                'classes': n_classes,    
            }
        },

        loss_fn = {
            'name': 'crossentropy' if n_classes > 1 else 'bcedice',
            'kwargs': {
                'ignore_index': 255
            }
        },

        lr=5e-4,
        n_epochs=10
    )

    return config