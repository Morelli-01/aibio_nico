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
        _, _, self.opt, self.scheduler = config_loader(self.config, self.dataset)
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
                wandb.log({"train_loss": loss.item(), "fold": fold_n})

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            acc, _, _ = evaluate(net, test_dataloader, self.device, epoch)
            wandb.log({"Accuracy": acc, "fold": fold_n})

            val_loss = validation(net=net, val_loader=test_dataloader, losser=self.losser, epoch=epoch)
            validation_loss_values += val_loss
            mean_loss = sum(val_loss) / len(val_loss) if val_loss else 0  # Division by zero paranoia

            wandb.log({"val_loss": mean_loss, "fold": fold_n})

            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()
                if isinstance(last_lr, list) and len(last_lr) == 1:
                    last_lr = last_lr[0]
                    wandb.log({"lr": last_lr, "fold": fold_n})
                elif isinstance(last_lr, list) and len(last_lr) > 1:
                    for i, lr in enumerate(last_lr):
                        wandb.log({f"lr_{i}": lr, "fold": fold_n})

        return training_loss_values, validation_loss_values

    def train(self):
        # ============= Preparing dataset... ==================
        train_workers = self.config["train_workers"]
        evaluation_workers = self.config["evaluation_workers"]
        device = self.device
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        print('Creating the subsets of the dataset')

        self.init_wandb()
        folds_accuracies = []
        indices = np.arange(len(self.dataset))
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):

            net = copy.deepcopy(self.net).to(self.device)
            train_subset = Subset(self.dataset, train_idx)
            test_subset = Subset(self.dataset, val_idx)

            train_dataloader = DataLoader(train_subset, batch_size=self.config["batch_size"],
                                          num_workers=train_workers, drop_last=False, prefetch_factor=8, persistent_workers=True, collate_fn=self.collate, pin_memory=True)
            test_dataloader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=True,
                                         num_workers=evaluation_workers, drop_last=False, prefetch_factor=8, collate_fn=self.collate)

            self.train_loop_(net, train_dataloader, test_dataloader, fold)
            acc, _, _ = evaluate(net, test_dataloader, self.device, 0)
            wandb.config.update({f"fold-{fold}-acc": f"{acc.item():.3f}"})
            folds_accuracies.append(acc.item())

        folds_accuracies = torch.tensor(folds_accuracies)

        wandb.config.update({
            "acc_std": f"{torch.std(folds_accuracies, dim=0).item():.3f}",
            "acc_mean": f"{torch.mean(folds_accuracies, dim=0).item():.3f}",
        })
        wandb.finish(0)
