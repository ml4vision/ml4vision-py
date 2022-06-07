from .segm_config import get_segm_config
from .det_config import get_det_config

def get_config(client, dataset_name, dataset_owner=None):

    # load dataset info
    dataset = client.get_dataset_by_name(dataset_name, owner=dataset_owner)

    if dataset.annotation_type == 'SEGMENTATION':
        return get_segm_config(dataset)
    elif dataset.annotation_type == 'BBOX':
        return get_det_config(dataset)
    else:
        raise RuntimeError(f'Trainer not implemented for dataset of annotation type: {dataset.annotation_type}.')
