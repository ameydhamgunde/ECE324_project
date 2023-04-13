from torch.optim import AdamW
from omegaconf import OmegaConf
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Config
from training_code import MidiMixIterDataset
from torch.utils.data import DataLoader
from models.t5 import T5ForConditionalGeneration
import torch.nn as nn
from utils import get_cosine_schedule_with_warmup, get_result_dir
import torch
import json
import pytorch_lightning as pl
import os
pl.seed_everything(365)


class MT3Net(pl.LightningModule):

    def __init__(self, config, model_config_path='config/mt3_config.json', result_dir='./results'):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        with open(model_config_path) as f:
            config_dict = json.load(f)
        config = T5Config.from_dict(config_dict)
        self.model: nn.Module = T5ForConditionalGeneration(config)
        
        if self.config.get('pretrained', None) is not None:
            self.model.load_state_dict(torch.load(self.config.pretrained, map_location='cpu'), strict=True)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('train_loss', outputs.loss)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('val_loss', outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.lr)
        warmup_step = int(self.config.num_training_steps / 100)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_step, num_training_steps=self.config.num_training_steps),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

    def train_dataloader(self):
        train_path = ('C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/train_data')
        dataset = MidiMixIterDataset(train_path, mel_length=self.config.mel_length, event_length=self.config.event_length, **self.config.data.config)
        train_loader = DataLoader(
            dataset, batch_size=self.config.per_device_batch_size, num_workers=0, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        test_path = ('C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/val_data')
        dataset = MidiMixIterDataset(test_path, mel_length=self.config.mel_length, event_length=self.config.event_length, **self.config.data.config)
        val_loader = DataLoader(
            dataset, batch_size=self.config.per_device_batch_size, num_workers=0, pin_memory=True)
        return val_loader


def main(config, model_config, result_dir):
    model = MT3Net(config, model_config, result_dir)
    model2 = MT3Net.load_from_checkpoint('C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/results/001/version_0/checkpoints/epoch=50-step=2550.ckpt', config=config, model_config=model_config, result_dir=result_dir)

    model3 = MT3Net.load_from_checkpoint('C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/results/001/version_0/checkpoints/last.ckpt', config=config, model_config=model_config, result_dir=result_dir)

    #print(model)
    logger = TensorBoardLogger(save_dir='/'.join(result_dir.split('/')[:-1]),
                               name=result_dir.split('/')[-1])

    num_training_steps = int(config['num_training_steps'])
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', save_last=True, save_top_k=5, save_weights_only=True)
    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        default_root_dir=os.path.join(os.getcwd(), 'logs'),
        callbacks=[lr_monitor, checkpoint_callback],
        precision=32,
        max_steps=num_training_steps,
        limit_train_batches=50, limit_val_batches=40,
        accelerator='cpu',
        accumulate_grad_batches=config.grad_accum)
    
    # trainer.fit(model)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Untrained model:")
    trainer.validate(model)
    print("Epoch 50, step 2550")
    trainer.validate(model2)
    print("Last Checkpoint")
    trainer.validate(model3)


if __name__ == "__main__":
    p = os.getcwd()
    print(p)
    conf_file = 'C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/config/config.yaml'
    model_config = 'C:/Users/ameyd/Documents/Assignments/ece324/mt3-pytorch/config/mt3_config.json'
    print(f'Config {conf_file}')
    conf = OmegaConf.load(conf_file)
    result_dir = get_result_dir()
    print('Creating: ', result_dir)
    os.makedirs(result_dir, exist_ok=False)
    shutil.copy(conf_file, f'{result_dir}/config.yaml')
    main(conf, model_config, result_dir)
