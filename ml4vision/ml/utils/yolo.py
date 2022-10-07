import os
import glob

def upload_latest_model(project, run_location='./runs/train', size=640, nms_threshold=0.45):
    # find latest version
    exp_path = sorted(glob.glob(os.path.join(run_location, '*')))[-1]
    result_file = os.path.join(exp_path, 'best.csv')
    with open(result_file, 'r') as f:
        result = f.readlines()[1]
        mp, mr, map50, map, conf = [float(item) for item in result.split(',')]

    # upload model
    print('Uploading model to ml4vision ...')
    remote_model = project.client.get_or_create_model(
        f"{project.name}-model",
        description = '',
        project = project.uuid,
        categories = project.categories,
        annotation_type = "BBOX",
        architecture = "object_detection_fn"
    )

    remote_model.add_version(
        os.path.join(exp_path, 'weights', 'best.pt'),
        params = {
            'model_type': 'yolov5',
            'min_size': size,
            'threshold': conf,
            'nms_threshold': nms_threshold
        },
        metrics = {
            'map50': round(map50, 3),
            'map': round(map, 3),
            'precision': round(mp, 3),
            'recall': round(mr, 3)
        }
    )