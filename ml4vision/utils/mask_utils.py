from ml4vision.utils.colormap import get_colormap
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


def annotations_to_mask(annotations, size):

    w, h = size
    label = np.zeros((h, w), dtype=np.uint8)
    
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        mask = ann_to_mask(ann['segmentation'], (w,h))
        label[y:y+h,x:x+w] += mask * (i+1)
    
    mask = Image.fromarray(label, mode='P')
    mask.putpalette(get_colormap())

    return mask

def empty_mask(size):
    return Image.new('L',size)