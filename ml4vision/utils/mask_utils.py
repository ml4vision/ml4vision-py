from ml4vision.utils.colormap import get_colormap
from ml4vision.utils.rle_cython import compute_rle
from PIL import Image
import numpy as np

def ann_to_mask(ann, size):
    w, h = size
    mask = np.zeros(w * h, dtype=np.uint8)

    index = 0
    zeros = True
    for count in ann['rle']:
        if not zeros:
            mask[index : index + count] = 1
        index+=count
        zeros = not zeros

    return np.reshape(mask, [h, w])


def annotations_to_label(annotations, size):

    w, h = size
    label = np.zeros((h, w), dtype=np.uint8)
    
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        mask = ann_to_mask(ann['segmentation'], (w,h))
        label[y:y+h,x:x+w] += mask * (i+1)
    
    label = Image.fromarray(label, mode='P')
    label.putpalette(get_colormap())

    return label

def get_rle(mask):
    counts = compute_rle(mask.flatten())
    size = [mask.shape[1], mask.shape[0]]
    return {"counts": counts, "size": size}

def label_to_annotations(label):

    annotations = []

    # loop over unique instances
    instance_ids = np.unique(label)
    instance_ids = instance_ids[instance_ids != 0]

    for id in instance_ids:
        mask = (label == id)
        
        y, x = np.nonzero(mask)
        if len(y) > 0:
            x_min, x_max, y_min, y_max = int(np.min(x)), int(np.max(x)), int(np.min(y)), int(np.max(y))
            cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
            rle = get_rle(cropped_mask)

            annotations.append(
                {
                    'segmentation': {
                        'rle': rle['counts'],
                        'size': rle['size']
                    },
                    'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                    'category_id': 0
                }
            )

    return annotations

def empty_mask(size):
    return Image.new('L',size)