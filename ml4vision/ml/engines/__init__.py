from .detection_engine import DetectionEngine
from .segmentation_engine import SegmentationEngine

def get_engine(config):
    if config['task'] == 'detection':
        return DetectionEngine(config)
    elif config['task'] == 'segmentation':
        return SegmentationEngine(config)
    else:
        RuntimeError(f'Engine of type {config["task"]} is not available.')