from .node import Node

def get_segm_config(dataset):

    client = dataset.client
    dataset_name = dataset.name
    dataset_owner = dataset.owner['username']   
    categories = dataset.categories

    binary = len(categories) == 1

    cfg = Node(
        dict(
            client = client,

            dataset_info = dict(
                name = dataset_name,
                owner = dataset_owner,
                categories = categories
            ),

            task = 'segmentation',

            save = True,
            save_location = './output',

            display = True,
            display_it = 50,

            pretrained_model = None,

            train_dataset = Node(dict(
                name = 'segmentation',
                params = Node(dict(
                    client = client,
                    name = dataset_name,
                    owner = dataset_owner,
                    labeled_only = True,
                    approved_only = False,
                    split = True,
                    train = True,
                    cache_location = './dataset',
                    min_size = 1000,
                    ignore_zero = False if binary else True,
                )),
                batch_size = 4,
                num_workers = 4
            )),

            val_dataset = Node(dict(
                name = 'segmentation',
                params = Node(dict(
                    client = client,
                    name = dataset_name,
                    owner = dataset_owner,
                    labeled_only = True,
                    approved_only = False,
                    split = True,
                    train = False,
                    cache_location = './dataset',
                    ignore_zero = False if binary else True,
                )),
                batch_size = 1,
                num_workers = 4
            )),

            model = Node(dict(
                name = 'unet',
                params = Node(dict(
                    encoder_name = 'resnet18',
                    classes = len(categories),
                )),
            )),

            loss = Node(dict(
                name = 'bcedice' if binary else 'crossentropy',
                params = Node(dict(
                    ignore_index = 255
                ))
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