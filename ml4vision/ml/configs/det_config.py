from .node import Node
from ..utils.centernet.mapping import mapping as centernet_mapping

def get_det_config(project):

    client = project.client
    project_name = project.name
    project_owner = project.owner['username']   
    categories = project.categories

    cfg = Node(
        dict(
            client = client,

            project_info = dict(
                name = project_name,
                owner = project_owner,
                categories = categories
            ),

            task = 'detection',

            save = True,
            save_location = './output',

            display = True,
            display_it = 50,

            pretrained_model = None,

            train_dataset = Node(dict(
                name = 'detection',
                params = Node(dict(
                    client = client,
                    name = project_name,
                    owner = project_owner,
                    labeled_only = True,
                    approved_only = False,
                    split = True,
                    train = True,
                    cache_location = './dataset',
                    min_size = 1000,
                    mapping = centernet_mapping
                )),
                batch_size = 4,
                num_workers = 4
            )),

            val_dataset = Node(dict(
                name = 'detection',
                params = Node(dict(
                    client = client,
                    name = project_name,
                    owner = project_owner,
                    labeled_only = True,
                    approved_only = False,
                    split = True,
                    train = False,
                    cache_location = './dataset',
                    mapping = centernet_mapping
                )),
                batch_size = 1,
                num_workers = 4
            )),

            model = Node(dict(
                name = 'unet',
                params = Node(dict(
                    encoder_name = 'resnet18',
                    classes = 3 + (len(categories) if len(categories) > 1 else 0)
                )),
                init_output = True
            )),

            loss = Node(dict(
                name = 'centernet',
                params = Node(dict(
                )),
            )),

            solver = Node(dict(
                lr = 5e-4,
                num_epochs = 10
            )),

            transform = Node(dict(
                resize = True,
                min_size = 512,
                random_crop = True,
                crop_size = 256,
                flip_horizontal = True,
                flip_vertical = True,
                random_brightness_contrast = True,
            ))
        )
    )

    return cfg