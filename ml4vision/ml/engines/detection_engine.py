from .engine import Engine
from ..utils.collate import box_collate_fn
from ..utils.visualizer import ObjectDetectionVisualizer
from ..utils.centernet.parse_detections import parse_detections, parse_batch_detections
from ..utils.ap_metrics import APMeter
import torch
import os
import json


class DetectionEngine(Engine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set visualizer
        self.visualizer = ObjectDetectionVisualizer()

        # set custom collate function
        self.train_dataset_it.collate_fn = box_collate_fn
        self.val_dataset_it.collate_fn = box_collate_fn

    def reset_metrics(self):
        self.ap_meter = APMeter()

    def compute_metrics(self, pred, sample):
        pred_boxes, pred_scores = parse_batch_detections(pred, threshold=0.1)
        for pred, scores, gt in zip(pred_boxes, pred_scores, sample['boxes']):
            self.ap_meter.add_detections(pred, scores, gt)

    def get_metrics(self):
        ap, ap_dict = self.ap_meter.get_ap()
        return {
            'map': ap,
            'working_point': ap_dict
        }

    def display(self, pred, sample):
        image = sample['image'][0]
        pred_boxes, _ = parse_detections(pred[0], threshold=0.1)
        gt_boxes = sample['boxes'][0]

        self.visualizer.display(image, pred_boxes.cpu(), gt_boxes)

    def forward(self, sample):
        images = sample['image'].to(self.device)
        heatmap = sample['heatmap'].to(self.device)
        scalemap = sample['scalemap'].to(self.device)
        classmap = sample['classmap'].to(self.device)

        pred = self.model(images)
        loss = self.loss_fn(pred, heatmap, scalemap, classmap)
        
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
