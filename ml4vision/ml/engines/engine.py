import os
import shutil
import torch
from datasets import get_dataset
from models import get_model
from losses import get_loss
from utils.meters import AverageMeter
from tqdm import tqdm

class Engine:

    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.setup()

    def setup(self):
        cfg = self.config

        # create output dir
        if cfg['save']:
            os.makedirs(cfg['save_dir'], exist_ok=True)

        self.train_dataset_it, self.val_dataset_it = self.get_dataloaders()
        self.model = self.get_model()
        self.loss_fn = self.get_loss()
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()

    def get_dataloaders(self):
        cfg = self.config

        train_dataset = get_dataset(
            cfg['train_dataset']['name'], cfg['train_dataset']['kwargs'])
        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=cfg['train_dataset']['workers'], pin_memory=True if self.device.type == 'cuda' else False)

        # val dataloader
        val_dataset = get_dataset(
            cfg['val_dataset']['name'], cfg['val_dataset']['kwargs'])
        val_dataset_it = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg['val_dataset']['batch_size'], shuffle=False, drop_last=False, num_workers=cfg['val_dataset']['workers'], pin_memory=True if self.device.type == 'cuda' else False)

        return train_dataset_it, val_dataset_it

    def get_model(self):
        cfg = self.config

        model = get_model(cfg['model']['name'], cfg['model']['kwargs'], cfg['model'].get('init_output')).to(self.device)

        # load checkpoint
        if cfg['model_path'] is not None and os.path.exists(cfg['model_path']):
            print(f'Loading model from {cfg["model_path"]}')
            state = torch.load(cfg['model_path'])
            model.load_state_dict(state['model_state_dict'], strict=True)

        return model

    def get_loss(self):
        cfg = self.config

        loss_fn = get_loss(cfg['loss_fn']['name'], cfg['loss_fn'].get('kwargs', {}))

        return loss_fn

    def get_optimizer_and_scheduler(self):
        cfg = self.config

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])

        lr_lambda = lambda epoch: (1 - (epoch/cfg['n_epochs'])) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return optimizer, scheduler

    def get_metrics(self):
        return {}

    def reset_metrics(self):
        pass

    def compute_metrics(self, pred, sample):
        pass

    def display(self, pred, sample):
        pass

    def forward(self, sample):
        pass

    def trace_model(self):
        pass

    def train_step(self):
        cfg = self.config

        # define meters
        loss_meter = AverageMeter()
 
        self.model.train()
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            pred, loss = self.forward(sample)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if cfg['display'] and i % cfg['display_it'] == 0:
                with torch.no_grad():
                    self.display(pred, sample)

        return loss_meter.avg

    def val_step(self):
        cfg = self.config

        self.reset_metrics()
        
        # define meters
        loss_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):
                pred, loss = self.forward(sample)
                loss_meter.update(loss.item())

                if cfg['display'] and i % cfg['display_it'] == 0:
                    self.display(pred, sample)

                self.compute_metrics(pred, sample)

        metrics = self.get_metrics()

        return loss_meter.avg, metrics

    def save_checkpoint(self, epoch, best_val=False, best_val_loss=0, metrics={}):
        cfg = self.config

        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }

        print("=> saving checkpoint")
        file_name = os.path.join(cfg['save_dir'], "checkpoint.pth")
        torch.save(state, file_name)
       
        if best_val:
            print("=> saving best_val checkpoint")
            shutil.copyfile(
                file_name, os.path.join(cfg['save_dir'], "best_val_model.pth")
            )

    def train(self):
        cfg = self.config
    
        best_val_loss = float('inf')
        for epoch in range(cfg['n_epochs']):
            print(f'Starting epoch {epoch}')

            train_loss = self.train_step()
            val_loss, metrics = self.val_step()

            print(f'==> train loss: {train_loss}')
            print(f'==> val loss: {val_loss}')
            print(f'metrics: {metrics}')

            best_val = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            self.save_checkpoint(epoch, best_val=best_val, best_val_loss=best_val_loss, metrics=metrics)

            self.scheduler.step()

        self.trace_model()

