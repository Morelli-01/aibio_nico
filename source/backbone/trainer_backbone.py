import os, sys, wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from source import utils
from source.utils import *
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch, random
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.model_selection import KFold
import copy


class Trainer():
    def __init__(self, net, device, config, opt, losser, collate, scheduler=None):
        self.net = net.to(device)
        self.device = device
        self.config = config
        self.opt = opt
        self.losser = losser
        self.scheduler = scheduler
        self.collate = collate
        # self.seed = random.randint(1, 10000)
        self.seed = 333
        self.gen = torch.Generator().manual_seed(self.seed)

    # def load_checkpoint(self):
    #     checkpoint = load_weights(self.config['load_checkpoint'], self.net, self.device)
    #     self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    #     if checkpoint['scheduler_state_dict'] is not None:
    #         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    #     last_epoch = checkpoint['epoch']
    #     training_loss_values = checkpoint['training_loss_values']
    #     validation_loss_values = checkpoint['validation_loss_values']
    #     self.config['batch_size'] = checkpoint['batch_size']
    #     return (last_epoch, training_loss_values, validation_loss_values)

    def init_wandb(self):
        assert wandb.api.api_key, "the api key has not been set!\n"
        # print(f"wandb key: {wandb.api.api_key}")
        wandb.login(verify=True)
        wandb.init(
            project=self.config['project_name'],
            name=self.config['run_name'],
            config=self.config
        )
        wandb.config.update({"seed": self.seed})

    def train_loop_(self, train_dataloader):
        # ============= Loading full checkpoint ==================
        # if self.config['load_checkpoint'] is not None:
        #     assert self.config['backbone_weights'] == None, "Config conflict: can't load a checkpoint and backbone weights."
        #     assert self.config['head_weights'] == None, "Config conflict: can't load a checkpoint and head weights."
        #     print('Loading latest checkpoint... ')
        #     last_epoch, training_loss_values, validation_loss_values = self.load_checkpoint()
        #     print(f"Checkpoint {self.config['load_checkpoint']} Loaded")
        # else:
        last_epoch = 0
        training_loss_values = []  # store every training loss value
        validation_loss_values = []  # store every validation loss value

        # ============= Training Loop ===================

        self.net = self.net.to(self.device)
        if self.config['multiple_gpus']:
            self.net = nn.DataParallel(self.net)
        print("Starting training...", flush=True)
        stream_ = torch.cuda.Stream()
        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            self.net.train()

            for i, (x_batch, y_batch, metadata) in enumerate(train_dataloader):

                loss = self.losser(self.device, (x_batch, y_batch, metadata), self.net, stream_)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.opt.step()
                training_loss_values.append(loss.item())
                wandb.log({"train_loss": loss.item()})

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()
                if isinstance(last_lr, list) and len(last_lr) == 1:
                    last_lr = last_lr[0]
                    wandb.log({"lr": last_lr})
                elif isinstance(last_lr, list) and len(last_lr) > 1:
                    for i, lr in enumerate(last_lr):
                        wandb.log({f"lr_{i}": lr})

        return training_loss_values, validation_loss_values

    def train(self, dataset):
        # ============= Preparing dataset... ==================
        self.dataset = dataset
        train_workers = self.config["train_workers"]
        device = self.device

        train_dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True,
                                      num_workers=train_workers, drop_last=True, prefetch_factor=10, persistent_workers=True, collate_fn=self.collate,
                                      pin_memory=True)
        self.init_wandb()

        self.train_loop_(train_dataloader)

        name = str(self.config['checkpoint_dir']) + "/" + str(self.config['run_name'])
        torch.save(
            {
                "epoch": self.config['epochs'],
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "batch_size": self.config['batch_size'],
                "optimizer": self.config['opt'],
                "scheduler_state_dict": self.scheduler.state_dict() if (self.scheduler is not None) else None,
                "seed": self.seed
            },
            name
        )
        print(f"Model saved in {name}.")
        wandb.finish(0)
