import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
import torch.nn as nn
import torch, random, wandb, copy
import numpy as np
from sklearn.model_selection import KFold
from source.utils import *


class Trainer():
    def __init__(self, net, device, config, opt, dataset, collate, losser, scheduler=None):
        self.net = net.to(device)
        self.device = device
        self.config = config
        self.opt = opt
        self.scheduler = scheduler
        self.collate = None
        # self.seed = random.randint(1, 10000)
        self.seed = 333
        self.gen = torch.Generator().manual_seed(self.seed)
        self.dataset = dataset
        self.losser = losser

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

    def train_loop_(self, net, train_dataloader, test_dataloader, fold_n):
        # ============= Loading full checkpoint ==================
        last_epoch = 0
        training_loss_values = []  # store every training loss value
        validation_loss_values = []  # store every validation loss value

        # ============= Training Loop ===================
        self.opt, self.sched = load_opt(self.config, net, self.dataset)
        if self.config['multiple_gpus']:
            net = nn.DataParallel(net)
        print("Starting training...", flush=True)
        stream_ = torch.cuda.Stream()
        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")

            for i, (x_batch, y_batch, metadata) in enumerate(train_dataloader):
                net.train()
                loss = self.losser((x_batch, y_batch, metadata), net, stream_)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())
                wandb.log({"train_loss": loss.item()})

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                if self.config['sched'] == "onecycle":
                    self.scheduler.step()
                    last_lr = self.scheduler.get_last_lr()
                    wandb.log({"lr": last_lr[0]})

            if self.scheduler is not None and self.config['sched'] != "onecycle":
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()
                wandb.log({"lr": last_lr[0]})

            acc, _, _, val_loss = evaluate(net, test_dataloader, self.device, epoch, self.losser)
            wandb.log({"Accuracy": acc})
            # val_loss = validation(net=net, val_loader=test_dataloader, losser=self.losser, epoch=epoch)
            validation_loss_values += val_loss
            mean_loss = sum(val_loss) / len(val_loss) if val_loss else 0  # Division by zero paranoia

            wandb.log({"val_loss": mean_loss})

        return training_loss_values, validation_loss_values

    def train(self):
        # ============= Preparing dataset... ==================
        train_workers = self.config["train_workers"]
        evaluation_workers = self.config["evaluation_workers"]
        device = self.device
        print('Creating the subsets of the dataset')

        self.init_wandb()
        train_subset, test_subset = random_split(self.dataset, [0.7, 0.3])

        # net = copy.deepcopy(self.net).to(self.device)

        train_dataloader = DataLoader(train_subset, batch_size=self.config["batch_size"],
                                      num_workers=train_workers, drop_last=True, prefetch_factor=8, persistent_workers=False, collate_fn=self.collate)
        test_dataloader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=True,
                                     num_workers=evaluation_workers, drop_last=True, prefetch_factor=4, collate_fn=self.collate)

        self.train_loop_(self.net, train_dataloader, test_dataloader, 0)

        wandb.finish(0)
