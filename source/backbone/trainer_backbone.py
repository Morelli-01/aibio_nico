import os, sys, wandb
from zmq import device
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
from source.sampler import *
from source.net import DataAugmentationDINO


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
                if self.config['sched'] == "onecycle":
                    self.scheduler.step()
                    last_lr = self.scheduler.get_last_lr()
                    wandb.log({"lr": last_lr[0]})

            if self.scheduler is not None and self.config['sched'] != "onecycle":
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()
                wandb.log({"lr": last_lr[0]})

        return training_loss_values, validation_loss_values

    def train(self, dataset):
        # ============= Preparing dataset... ==================
        self.dataset = dataset
        train_workers = self.config["train_workers"]
        device = self.device
        self.losser.set_experiment_idx(dataset.experiment_idx)
        self.losser.set_cell_type_idx(dataset.cell_type_idx)

        if self.config["sampler"]:
            sampler = ExperimentSampler(dataset, batch_size=self.config["batch_size"], shuffle=True)
            train_dataloader = DataLoader(dataset,
                                          num_workers=train_workers, prefetch_factor=10, persistent_workers=True,
                                          batch_sampler=sampler)
        else:
            train_dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True,
                                          num_workers=train_workers, drop_last=True, prefetch_factor=10, persistent_workers=True)

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


class DinoTrainer():
    def __init__(self, net, device, config, opt, losser, collate, scheduler=None):
        self.student = net
        self.teacher = copy.deepcopy(net)
        self.teacher_warmup_epochs = 5
        self.DinoAug = DataAugmentationDINO(global_crops_scale=(
            0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8)
        self.embed_dim = 384
        self.head_out_dim = 65536
        self.device = device
        self.config = config
        self.opt = opt
        self.losser = DINOLoss(
            out_dim=self.head_out_dim,
            ncrops=10,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=0,
            nepochs=self.config["epochs"]
        ).to(device=self.device)
        self.scheduler = scheduler
        self.collate = collate
        # self.seed = random.randint(1, 10000)
        self.seed = 333
        self.gen = torch.Generator().manual_seed(self.seed)
        if self.config["load_backbone"] is not None:
            checkpoint = torch.load(self.config["load_backbone"])['model_state_dict']
            state_dict = {}
            for k in checkpoint.keys():
                if "backbone" not in k:
                    continue
                new_k = k.removeprefix("backbone.")
                state_dict[new_k] = checkpoint[k]
            msg = self.teacher.load_state_dict(state_dict)
            print(msg, f"Teacher weights loaded from checkpoint{self.config['load_backbone']} ")

    def get_params_groups(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

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
        last_epoch = 0
        training_loss_values = []  # store every training loss value
        validation_loss_values = []  # store every validation loss value

        # ============= Training Loop ===================

        if self.config['multiple_gpus']:
            self.student = nn.DataParallel(self.student)
            self.teacher = nn.DataParallel(self.teacher)
        print("Starting training...", flush=True)
        stream_ = torch.cuda.Stream()
        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            self.student.train()
            self.teacher.train()

            for i, (x_batch, y_batch, metadata) in enumerate(train_dataloader):

                with torch.cuda.stream(stream_):
                    images = self.DinoAug(x_batch)
                    images = [im.to(device=self.device, non_blocking=True) for im in images]

                teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = self.student(images)
                loss = self.losser(student_output, teacher_output, epoch)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())
                wandb.log({"train_loss": loss.item()})

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                with torch.no_grad():
                    m = self.scheduler.get_last_lr()[0]  # momentum parameter
                    for param_q, param_k in zip(self.student.module.parameters(), self.teacher.module.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                if self.config['sched'] == "onecycle":
                    self.scheduler.step()
                    last_lr = self.scheduler.get_last_lr()
                    wandb.log({"lr": last_lr[0]})

            if self.scheduler is not None and self.config['sched'] != "onecycle":
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()
                wandb.log({"lr": last_lr[0]})

            name = str(self.config['checkpoint_dir']) + "/" + str(self.config['run_name'])
            torch.save(
                {
                    "epoch": self.config['epochs'],
                    "model_state_dict": self.student.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "batch_size": self.config['batch_size'],
                    "optimizer": self.config['opt'],
                    "scheduler_state_dict": self.scheduler.state_dict() if (self.scheduler is not None) else None,
                    "seed": self.seed
                },
                name
            )
        return training_loss_values, validation_loss_values

    def train(self, dataset):
        # ============= Preparing dataset... ==================
        self.dataset = dataset
        train_workers = self.config["train_workers"]
        device = self.device

        self.student = utils.MultiCropWrapper(self.student, DINOHead(
            in_dim=384,
            out_dim=65536,
            use_bn=True,
            norm_last_layer=True,
        ))
        self.teacher = utils.MultiCropWrapper(
            self.teacher,
            DINOHead(in_dim=384,
                     out_dim=65536, use_bn=True),
        )
        self.teacher, self.student = self.teacher.to(self.device), self.student.to(self.device)
        if self.config["sampler"]:
            sampler = ExperimentSampler(dataset, batch_size=self.config["batch_size"], shuffle=True)
            train_dataloader = DataLoader(dataset,
                                          num_workers=train_workers, prefetch_factor=10, persistent_workers=True,
                                          batch_sampler=sampler)
        else:
            train_dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True,
                                          num_workers=train_workers, drop_last=True, prefetch_factor=10, persistent_workers=True)

        # ===========Load optim and sched =============================
        self.opt, self.scheduler = load_opt(self.config, self.student, self.dataset)
        # params_groups = utils.get_params_groups(self.student)
        # self.opt = torch.optim.AdamW(params_groups)  # to use with ViTs

        # lr_schedule = cosine_scheduler(
        #     base_value=0.0005 * (self.config["batch_size"]) / 256.,  # linear scaling rule
        #     final_value=1e-6,
        #     epochs=self.config["epochs"], niter_per_ep=len(train_dataloader),
        #     warmup_epochs=5
        # )
        # wd_schedule = cosine_scheduler(
        #     0.04,
        #     0.4,
        #     epochs=self.config["epochs"], niter_per_ep=len(train_dataloader)
        # )
        # # momentum parameter is increased to 1. during training with a cosine schedule
        # momentum_schedule = cosine_scheduler(0.9995, 1,
        #                                      epochs=self.config["epochs"], niter_per_ep=len(train_dataloader))
        # print(f"Loss, optimizer and schedulers ready.")
    # ======================================================

        self.init_wandb()

        self.train_loop_(train_dataloader)

        name = str(self.config['checkpoint_dir']) + "/" + str(self.config['run_name'])
        torch.save(
            {
                "epoch": self.config['epochs'],
                "model_state_dict": self.student.state_dict(),
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
