import torch
from .engine import Engine
from ..utils.visualizer import SegmentationVisualizer
from ..utils.iou_evaluator import IOUEvaluator
import os
import json


class SegmentationEngine(Engine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary = len(self.config.dataset_info['categories']) == 1
        self.visualizer = SegmentationVisualizer()

        n_classes = len(self.config.dataset_info['categories'])
        ignore_index = self.config.loss.params.get('ignore_index', -1)
        self.iou_meter = IOUEvaluator(n_classes + 1 if self.binary else n_classes,
                                ignore_index=ignore_index)

    def reset_metrics(self):
        self.iou_meter.reset()

    def compute_metrics(self, pred, sample):
        labels = sample['label'].to(self.device)
        labels = labels.unsqueeze(1).long()

        if self.binary:
            self.iou_meter.addBatch((pred > 0).long(), labels)
        else:
            self.iou_meter.addBatch(pred.argmax(dim=1,keepdim=True), labels)

    def get_metrics(self):
        if self.binary:
            iou = self.iou_meter.getIoU()[1][1] 
            return {'mean_iou': iou}
        else:
            miou, iou = self.iou_meter.getIoU()
            return {'mean_iou': miou, 'class_iou': iou}

    def display(self, pred, sample):
        cfg = self.config

        # display a single image
        image = sample['image'][0]
        pred = (pred[0] > 0).cpu() if self.binary else torch.argmax(pred[0],dim=0).cpu()
        gt = sample['label'][0]

        self.visualizer.display(image, pred, gt)

    def forward(self, sample):
        images = sample['image'].to(self.device)
        labels = sample['label'].to(self.device)

        if self.binary:
            labels = labels.unsqueeze(1).float()
        else:
            labels = labels.long()

        pred = self.model(images)
        loss = self.loss_fn(pred, labels)

        return pred, loss

    def trace_model(self):
        cfg = self.config

        # load best checkpoint
        model = self.model.to(torch.device('cpu'))
        state = torch.load(os.path.join(cfg.save_location, 'best_val_model.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'], strict=True)
        model.eval()

        traced_model = torch.jit.trace(model, torch.randn(1, 3, 512, 512))
        traced_model.save(os.path.join(cfg.save_location, 'best_val_model.pt'))

        with open(os.path.join(cfg.save_location, 'metrics.json'), 'w') as f:
            json.dump(state['metrics'], f)

        with open(os.path.join(cfg.save_location, 'categories.json'), 'w') as f:
            json.dump(cfg.dataset_info['categories'], f)