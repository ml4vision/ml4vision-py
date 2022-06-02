class Node(dict):

    def __init__(self, init_dict=dict()):
        super().__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

def get_default_config(client, dataset_name, dataset_owner=None):

    dataset = client.get_dataset_by_name(dataset_name, dataset_owner)
    
    if dataset.annotation_type == 'BBOX':
        task = 'detection'
    elif dataset.annotation_type == 'SEGMENTATION':
        task = 'segmentation'
    else:
        raise RuntimeError(f'Trainer not implemented for dataset of annotation type: {dataset.annotation_type}.')
    
    categories = dataset.categories

    cfg = Node(
        dict(
            client = client,

            task = task,

            save = True,
            save_location = './output',

            display = True,
            display_it = 50,

            weights = None,

            dataset = Node(dict(
                name = dataset_name,
                owner = dataset_owner,
                categories = categories,
                labeled_only = True,
                approved_only = False,
                cache_location = './dataset',
                min_size = 1000
            )),

            dataloader = Node(dict(
                train_batch_size = 4,
                val_batch_size = 1,
                train_num_workers = 4,
                val_num_workers = 4
            )),

            model = Node(dict(
                name = 'unet',
                kwargs = dict(
                    encoder_name = 'resnet18',
                    encoder_weights = 'imagenet'
                )
            )),

            solver = Node(dict(
                lr = 5e-4,
                num_epochs = 10
            )),

            transform = Node(dict(
                resize = True,
                min_size = 512,
                crop = True,
                crop_size = 256,
                flip_horizontal = True,
                flip_vertical = True
            ))
        )
    )

    return cfg