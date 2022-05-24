from .dataset import ObjectDetectionDataset, SegmentationDataset

def get_dataset(name, dataset_kwargs={}):
    if name == "segmentation":
        return SegmentationDataset(**dataset_kwargs)
    elif name == "detection":
        return ObjectDetectionDataset(**dataset_kwargs)
    else:
        raise RuntimeError(f'Dataset {name} is not available!') 