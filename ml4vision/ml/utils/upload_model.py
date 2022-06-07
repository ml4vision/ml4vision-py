import os
import json

def upload_model(config, model_name=None):

    model_name = model_name or f"{config.dataset_info['name']}-model"
    annotation_type = "BBOX" if config.task == 'detection' else "SEGMENTATION"
    architecture = 'object_detection_fn' if annotation_type == "BBOX" else "segmentation_fn"

    model = config.client.get_or_create_model(
        model_name,
        description='',
        dataset = config.client.get_dataset_by_name(config.dataset_info['name'], config.dataset_info['owner']).uuid,
        categories=config.dataset_info['categories'],
        annotation_type=annotation_type,
        architecture=architecture
    )

    with open(os.path.join(config.save_location, 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    print('adding version')
    if annotation_type == "BBOX":
            
        model.add_version(
            os.path.join(config.save_location, 'best_val_model.pt'),
            params = {
                'min_size': config.transform.min_size,
                'pad': 32,
                'normalize': True,
                'threshold': float(metrics["working_point"]["confidence"])
            }
        )

    else:
        model.add_version(
            os.path.join(config.save_location, 'best_val_model.pt'),
            params = {
                'min_size': config.transform.min_size,
                'pad': 32,
                'normalize': True,
            }
        )

