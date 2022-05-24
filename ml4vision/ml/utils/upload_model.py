import os
import json

def upload_model(client, config, model_name=None):

    model_name = model_name or f"{config['train_dataset']['kwargs']['name']}-model"
    annotation_type = "BBOX" if config["engine"] == 'detection' else "SEGMENTATION"
    architecture = 'object_detection_fn' if annotation_type == "BBOX" else "segmentation_fn"

    model = client.get_or_create_model(
        model_name,
        description='',
        categories=config["categories"],
        annotation_type=annotation_type,
        architecture=architecture
    )

    with open(os.path.join(config['save_dir'], 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    print('adding version')
    if annotation_type == "BBOX":
            
        model.add_version(
            os.path.join(config['save_dir'], 'best_val_model.pt'),
            params = {
                'min_size': 512,
                'pad': 32,
                'normalize': True,
                'threshold': float(metrics["working_point"]["confidence"])
            }
        )

    else:
        model.add_version(
            os.path.join(config['save_dir'], 'best_val_model.pt'),
            params = {
                'min_size': 512,
                'pad': 32,
                'normalize': True,
            }
        )

