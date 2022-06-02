import os
import random

from torch.utils.data import Dataset as TorchDataset
import json
import numpy as np

from ml4vision.ml.utils.image_utils import load_image
from PIL import Image


class ML4visionDataset(TorchDataset):
    def __init__(
        self,
        client=None,
        name='',
        owner=None,
        labeled_only=True,
        approved_only=False,
        split=False,
        train=True,
        cache_location="./dataset",
        min_size=None,
        transform=None,
        mapping=None
    ):

        dataset = client.get_dataset_by_name(name, owner=owner)
        dataset.load_samples(labeled_only=labeled_only, approved_only=approved_only)

        if split:
            n_samples = len(dataset.samples)
            dataset.samples = (
                dataset.samples[0 : int(n_samples * 0.8)]
                if train
                else dataset.samples[int(n_samples * 0.8) :]
            )

        dataset_loc = dataset.pull(location=cache_location)

        self.dataset = dataset
        self.dataset_loc = dataset_loc
        self.size = len(self.dataset.samples)
        self.min_size = min_size
        self.transform = transform
        self.mapping = mapping

    def __len__(self):
        return self.min_size or self.size

    def get_index(self, index):
        if self.min_size:
            index = random.randint(0, self.size - 1) if self.size > 1 else 0
        return index

    def get_image(self, sample):
        image_filename = sample.asset["filename"]
        image_path = os.path.join(self.dataset_loc, "images", image_filename)
        image = load_image(image_path)
        return image


class ObjectDetectionDataset(ML4visionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"ObjectDetectionDataset created, found {self.size} samples")

    def format_boxes(self, annotations, im_w, im_h):
        boxes = []
        category_ids = []
        for ann in annotations:

            x1, y1, w, h = ann["bbox"]
            x2, y2 = x1 + w - 1, y1 + h - 1

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            if x2 >= im_w:
                x2 = im_w - 1
            if y2 >= im_h:
                y2 = im_h - 1

            # remove corrupted boxes
            if x1 >= x2 or y1 >= y2 or w == 0 or h == 0:
                continue

            box = [x1, y1, x2, y2]
            boxes.append(box)
            category_ids.append(ann["category_id"] - 1)

        return boxes, category_ids

    def get_boxes(self, sample, im_w, im_h):
        label_filename = os.path.splitext(sample.asset["filename"])[0] + ".json"
        label_path = os.path.join(self.dataset_loc, "labels", label_filename)
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        boxes, category_ids = self.format_boxes(annotations, im_w, im_h)
        return boxes, category_ids

    def __getitem__(self, index):

        index = self.get_index(index)
        sample = self.dataset.samples[index]

        image = np.array(self.get_image(sample))
        im_h, im_w = image.shape[:-1]

        boxes, category_ids = self.get_boxes(sample, im_w, im_h)

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=boxes, category_ids=category_ids
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]

        sample = {
            "image": image,
            "boxes": boxes,
            "category_ids": category_ids,
        }

        if self.mapping:
            sample = self.mapping(sample)

        return sample


class SegmentationDataset(ML4visionDataset):
    def __init__(self, *args, ignore_zero=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_zero = ignore_zero
        print(f"SegmentationDataset created, found {self.size} samples")

    def get_label(self, sample):
        label_filename = os.path.splitext(sample.asset["filename"])[0] + ".png"
        label_path = os.path.join(self.dataset_loc, "labels", label_filename)
        label = np.array(Image.open(label_path))

        if self.ignore_zero:
            label = label - 1

        return label

    def __getitem__(self, index):
        index = self.get_index(index)
        sample = self.dataset.samples[index]

        image = np.array(self.get_image(sample))
        label = self.get_label(sample)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        sample = {
            "image": image,
            "label": label,
        }

        if self.mapping:
            sample = self.mapping(sample)

        return sample


class InstanceSegmentationDataset(ML4visionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"InstanceSegmentationDataset created, found {self.size} samples")

    def get_label(self, sample):
        cls_filename = os.path.splitext(sample.asset["filename"])[0] + "_cls.png"
        cls_path = os.path.join(self.dataset_loc, "labels", cls_filename)
        cls = np.array(Image.open(cls_path))

        inst_filename = os.path.splitext(sample.asset["filename"])[0] + "_inst.png"
        inst_path = os.path.join(self.dataset_loc, "labels", inst_filename)
        inst = np.array(Image.open(inst_path))

        return cls, inst

    def __getitem__(self, index):
        index = self.get_index(index)
        sample = self.dataset.samples[index]

        image = np.array(self.get_image(sample))
        cls_label, inst_label = self.get_label(sample)

        if self.transform:
            transformed = self.transform(image=image, masks=[cls_label, inst_label])
            image = transformed["image"]
            cls_label, inst_label = transformed["masks"]

        sample = {"image": image, "cls_label": cls_label, "inst_label": inst_label}

        if self.mapping:
            sample = self.mapping(sample)

        return sample
