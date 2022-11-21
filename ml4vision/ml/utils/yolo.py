import os
import glob
from ml4vision.client import MLModel

def upload_latest_model(project, run_location='./runs/train', size=640, nms_threshold=0.45):
    # find latest version
    exp_path = sorted(glob.glob(os.path.join(run_location, '*')))[-1]
    result_file = os.path.join(exp_path, 'best.csv')
    with open(result_file, 'r') as f:
        result = f.readlines()[1]
        mp, mr, map50, map, conf = [float(item) for item in result.split(',')]

    # create model
    if project.model is None:
        model = MLModel.create(
            project.client,
            f'{project.name}-model',
            project=project.uuid
        )
    else:
        model = project.model

    # add version
    model.add_version(
        os.path.join(exp_path, 'weights', 'best.pt'),
        categories=project.categories,
        architecture='object_detection_fn',
        params={
            'model_type': 'yolov5',
            'min_size': size,
            'threshold': conf,
            'nms_threshold': nms_threshold
        },
        metrics={
            'map50': round(map50, 3),
            'map': round(map, 3),
            'precision': round(mp, 3),
            'recall': round(mr, 3)
        }
    )