import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..utils.centernet.mapping import mapping as centernet_mapping

def get_det_config(config):

    engine_config = dict(
        categories = config.dataset.categories,

        save = config.save,
        save_dir = config.save_location,
    
        display = config.display,
        display_it = config.display_it,

        weights = config.weights,

        train_dataset = {
            'name': 'detection',
            'kwargs': {
                'client': config.client,
                'name': config.dataset.name,
                'owner': config.dataset.owner,
                'labeled_only': config.dataset.labeled_only,
                'approved_only': config.dataset.approved_only,
                'split': True,
                'train': True,
                'cache_location': config.dataset.cache_location,
                'min_size': config.dataset.min_size,
                'transform': get_train_transform(config),
                'mapping': centernet_mapping
            },
            'batch_size': config.dataloader.train_batch_size,
            'workers': config.dataloader.train_num_workers
        }, 

        val_dataset = {
            'name': 'detection',
            'kwargs': {
                'client': config.client,
                'name': config.dataset.name,
                'owner': config.dataset.owner,
                'labeled_only': config.dataset.labeled_only,
                'approved_only': config.dataset.approved_only,
                'split': True,
                'train': False,
                'cache_location': config.dataset.cache_location,
                'transform': get_val_transform(config),
                'mapping': centernet_mapping
            },
            'batch_size': config.dataloader.val_batch_size,
            'workers': config.dataloader.val_num_workers
        }, 

        model = {
            'name': config.model.name,
            'kwargs': dict(
                **config.model.kwargs,
                classes = 3 + (len(config.dataset.categories) if len(config.dataset.categories) > 1 else 0)
            ),
            'init_output': True
        },

        loss_fn = {
            'name': 'centernet',
        },

        lr = config.solver.lr,
        n_epochs = config.solver.num_epochs
    )

    return engine_config

def get_train_transform(config):
    transform_list = []
    
    if config.transform.resize:
        transform_list.append(A.SmallestMaxSize(max_size=config.transform.min_size))
    if config.transform.crop:
        crop_size = config.transform.crop_size
        min_size = crop_size - (crop_size * 0.15)
        max_size = crop_size + (crop_size * 0.15)
        transform_list.append(A.RandomSizedCrop([int(min_size),int(max_size)],config.transform.crop_size,config.transform.crop_size))
    if config.transform.flip_horizontal:
        transform_list.append(A.HorizontalFlip(p=0.5))
    if config.transform.flip_vertical:
        transform_list.append(A.VerticalFlip(p=0.5))
    
    transform_list.extend([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ])

    transform = A.Compose(
        transform_list, 
        bbox_params=A.BboxParams('pascal_voc', 
        label_fields=['category_ids'], 
        min_visibility=0.1)
    )

    return transform

def get_val_transform(config):
    transform = A.Compose([
        A.SmallestMaxSize(max_size=config.transform.min_size),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams('pascal_voc', label_fields=['category_ids'], min_visibility=0.1))

    return transform
