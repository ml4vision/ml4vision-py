import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..utils.centernet.mapping import mapping as centernet_mapping

def get_segm_config(config):

    engine_config = dict(
        categories = config.dataset.categories,

        save = config.save,
        save_dir = config.save_location,
    
        display = config.display,
        display_it = config.display_it,

        weights = config.weights,

        train_dataset = {
            'name': 'segmentation',
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
                'ignore_zero': True if len(config.dataset.categories) > 1 else False,
                'transform': get_train_transform(config),
            },
            'batch_size': config.dataloader.train_batch_size,
            'workers': config.dataloader.train_num_workers
        }, 

        val_dataset = {
            'name': 'segmentation',
            'kwargs': {
                'client': config.client,
                'name': config.dataset.name,
                'owner': config.dataset.owner,
                'labeled_only': config.dataset.labeled_only,
                'approved_only': config.dataset.approved_only,
                'split': True,
                'train': False,
                'cache_location': config.dataset.cache_location,
                'ignore_zero': True if len(config.dataset.categories) > 1 else False,
                'transform': get_val_transform(config),
            },
            'batch_size': config.dataloader.val_batch_size,
            'workers': config.dataloader.val_num_workers
        }, 

        model = {
            'name': config.model.name,
            'kwargs': dict(
                **config.model.kwargs,
                classes = len(config.dataset.categories)
            )
        },

        loss_fn = {
            'name': 'crossentropy' if len(config.dataset.categories) > 1 else 'bcedice',
            'kwargs': {
                'ignore_index': 255
            }
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
        min_size = crop_size - crop_size * 0.15
        max_size = crop_size + crop_size * 0.15
        transform_list.append(A.RandomSizedCrop([int(min_size),int(max_size)],config.transform.crop_size,config.transform.crop_size))
    if config.transform.flip:
        transform_list.append(A.Flip(p=0.5))
    
    transform_list.extend([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ])

    transform = A.Compose(transform_list)

    return transform

def get_val_transform(config):
    transform = A.Compose([
        A.SmallestMaxSize(max_size=config.transform.min_size),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ])

    return transform
