from .default_detection_cfg import get_config as get_detection_cfg
from .default_segmentation_cfg import get_config as get_segmentation_cfg

def get_config(client, dataset_name, dataset_owner=None):

    # get dataset
    dataset = client.get_dataset_by_name(dataset_name, dataset_owner)

    if dataset.annotation_type == 'BBOX': 
        config = get_detection_cfg(client, dataset_name, dataset_owner, categories=dataset.categories)
    elif dataset.annotation_type == 'SEGMENTATION':
        config = get_segmentation_cfg(client, dataset_name, dataset_owner, categories=dataset.categories)
    else:
        raise RuntimeError(f'Trainer not implemented for dataset of annotation type: {dataset.annotation_type}.')

    return config