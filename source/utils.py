from importlib import metadata
import os, sys

from matplotlib.pyplot import step

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.net import *
import torchvision.transforms.v2 as transforms
from prettytable import PrettyTable
import torch, wandb
import torch.nn.functional as F
import torchvision.transforms.v2.functional as transform_F
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Tuple
import yaml, random, math
from pathlib import Path
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import numpy as np
# import source.vision_transformer as vits


def load_yaml(inFile=sys.argv[1]) -> dict:
    """Loads a YAML configuration file from the first command-line argument.

    This function reads a YAML file specified as a command-line argument,
    parses its contents into a dictionary, and validates required paths.

    Usage:
        python my_script.py config.yaml

    Raises:
        AssertionError: If `checkpoint_dir` is not a valid directory.
        AssertionError: If `dataset_dir` is not a valid directory.
        AssertionError: If `load_checkpoint` is provided but not a valid file.

    Returns:
        dict: A dictionary containing the parsed YAML configuration.
    """
    with open(inFile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    display_configs(config)

    assert Path(config['checkpoint_dir']).is_dir(), "Please provide a valid directory to save checkpoints in."
    assert Path(config['dataset_dir']).is_dir(), "Please provide a valid directory to load dataset."
    if config['load_checkpoint'] is not None:
        assert Path(config['load_checkpoint']).is_file(), "Please provide a valid file to load checkpoint."

    return config


def display_configs(configs):
    t = PrettyTable(["Name", "Value"])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def load_device(config: dict):
    """Loads and returns the appropriate computing device based on the configuration.

    This function checks the "device" key in the given config dictionary.
    If "gpu" is specified, it ensures that CUDA is available and selects the first GPU.
    Otherwise, it defaults to the CPU.

    Args:
        config (dict): A dictionary containing a "device" key with values "gpu" or "cpu".

    Raises:
        AssertionError: If "gpu" is requested but CUDA is not available.

    Returns:
        str: The device identifier, either "cuda:0" for GPU or "cpu".
    """
    if config["device"] == "gpu":
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = "cuda:0"
        print(
            "Training network on {}({})".format(torch.cuda.get_device_name(device=device), torch.cuda.device_count())
        )

    else:
        device = torch.device("cpu")
    return device


def config_loader(config, dataset=None):
    options = {}
    if config['num_classes'] is not None:
        options['num_classes'] = config['num_classes']
    if config['epochs'] is not None:
        options['epochs'] = config['epochs']
    if config['backbone_name'] is not None:
        options['backbone_name'] = config['backbone_name']
    if config['device'] is not None:
        options['device'] = config['device']
    net = load_net(config["net"], options)

    loss = load_loss(config["loss"], options)
    opt, sched = load_opt(config, net, dataset)
    return (net, loss, opt, sched)


def load_net(netname: str, options={}) -> torch.nn.Module:
    if netname == "simclr":
        assert "backbone_name" in options.keys(), "The backbone_name option was not provided!"
        return SimCLR(backbone=options["backbone_name"], num_classes=options["num_classes"])
    if netname == "vit_base":
        return vit_base()
    elif netname == "vit_tiny":
        return vit_tiny()
    elif netname == "vit_small":
        return vit_small()
    elif netname == "vit_large":
        return vit_large()
    else:
        raise ValueError("Invalid netname")


def load_loss(lossname: str, options={}) -> Callable:
    if lossname == "simclrloss":
        return SimCLRLoss()
    if lossname == "ClassificationLoss":
        return ClassificationLoss(device=torch.device("cuda:0" if options['device'] == "gpu" else "cpu"))
    else:
        raise ValueError("Invalid lossname")


def load_opt(config: dict, net: torch.nn.Module, dataset: torch.utils.data.Dataset) -> torch.optim.Optimizer:
    if config['opt'] == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=float(config['lr']))
    elif config['opt'] == "adamW":
        opt = torch.optim.AdamW(net.parameters(), lr=float(config['lr']))
    else:
        raise ValueError("Invalid optimizer")

    sched = None
    if config["sched"] is None or config["sched"] == "poly":
        sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=config['epochs'], power=config['sched_pow'])
    elif config["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(config['lr']),
            steps_per_epoch=(len(dataset) // int(config["batch_size"])),
            epochs=config["epochs"]
        )
    elif config["sched"] == "CosineAnnealingWarmRestarts":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2, eta_min=1e-6)

    return opt, sched


def load_backbone(config: dict, net: torch.nn.Module):

    if config["load_backbone"] is not None:
        backbone_path = config["load_backbone"]
        checkpoint = torch.load(backbone_path, weights_only=False)
        student_weights = checkpoint['student']
        wht = {}
        if "vit" not in config["net"]:
            for k in list(student_weights.keys()):
                if "mlp" in k or "last_layer" in k:
                    continue
                new_key = k.removeprefix("module.backbone.backbone.")
                wht[new_key] = student_weights[k]
            net.backbone.load_state_dict(wht)
        else:
            for k in list(student_weights.keys()):
                if "head" in k:
                    continue
                new_key = k.removeprefix("module.backbone.")
                wht[new_key] = student_weights[k]
            net[0].load_state_dict(wht)

        print(f"Backbone weights {backbone_path} loaded!!!")

    else:
        print("No backbone to load!")

    return net


def baseline_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable, epochs=0):
    x_batch, y_batch, metadata = data

    x_batch, y_batch = x_batch.to(device), y_batch.to(device).type(torch.int64)
    out_feat = net.forward(x_batch).type(torch.float)
    loss = loss_func(out_feat, y_batch)
    return loss


def validation(net, val_loader, losser, epoch) -> Tuple[List[float], float]:
    """
    Computes the validation loss for a given neural network model.

    Parameters:
    - net: The neural network model.
    - val_loader: DataLoader for the validation dataset.
    - device: The device (CPU/GPU) where computations will be performed.
    - loss_func: The loss function used for evaluation.
    - losser: A function that computes the loss given the model, inputs, and loss function.
    - epoch: The current epoch number (used for logging).

    Returns:
    - validation_loss_values: A list of loss values for each validation batch.
    - accuracy: The total accuracy calculated on the whole testing set
    """

    validation_loss_values = []  # List to store loss values for each batch

    # Create a progress bar for validation
    pbar = tqdm(total=len(val_loader), desc=f"validation-{epoch+1}")

    net.eval()
    with torch.no_grad():
        stream_ = torch.cuda.Stream()
        for x_batch, y_batch, metadata in val_loader:  # Iterate through validation data
            # Compute loss using the provided 'losser' function
            loss = losser((x_batch, y_batch, metadata), net, stream_)

            validation_loss_values.append(loss.item())

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Validation Loss": loss.item()})

    return validation_loss_values  # Return the collected validation loss values


def evaluate(net, test_loader, device, epoch, losser):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        net.eval()
        validation_loss_values = []
        stream_ = torch.cuda.Stream()
        for x, y, metadata in tqdm(test_loader, desc=f"Eval-{epoch+1}"):
            validation_loss_values.append(losser((x, y, metadata), net, stream_).item())
            with torch.cuda.stream(stream_):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = net(x)

            if len(pred.shape) < 2:
                pred = pred.argmax(dim=0)
                predictions.append(torch.unsqueeze(pred, 0))
            else:
                pred = pred.argmax(dim=1)
                predictions.append(pred)
            labels.append(y)

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = torch.where(predictions == labels, 1, 0).sum() / len(labels)

    return acc, predictions, labels, validation_loss_values


def load_weights(checkpoint_path: str, net: torch.nn.Module, device: torch.cuda.device) -> torch.utils.checkpoint:
    """!Load only network weights from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if list(checkpoint['model_state_dict'])[0].__contains__('module'):
        model_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    else:
        model_dict = checkpoint['model_state_dict']
    net.load_state_dict(model_dict)
    return checkpoint


def save_model(epoch, net, opt, train_loss, val_loss, batch_size, checkpoint_dir, optimizer, seed, scheduler=None, run_name=None):
    name = os.path.join(checkpoint_dir, f"{run_name}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "training_loss_values": train_loss,
            "validation_loss_values": val_loss,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "scheduler_state_dict": scheduler.state_dict() if (scheduler is not None) else None,
            "seed": seed
        },
        name
    )
    print(f"Model saved in {name}.")

    # class GaussianBlur(object):
    #     """
    #     Apply Gaussian Blur to the PIL image.
    #     """

    #     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
    #         self.prob = p
    #         self.radius_min = radius_min
    #         self.radius_max = radius_max

    #     def __call__(self, img):
    #         do_it = random.random() <= self.prob
    #         if not do_it:
    #             return img

    #         return img.filter(
    #             ImageFilter.GaussianBlur(
    #                 radius=random.uniform(self.radius_min, self.radius_max)
    #             )
    #         )

    # class Solarization(object):
    #     """
    #     Apply Solarization to the PIL image.
    #     """

    #     def __init__(self, p):
    #         self.p = p

    #     def __call__(self, img):
    #         if random.random() < self.p:
    #             return ImageOps.solarize(img)
    #         else:
    #             return img


class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value

    def __call__(self, iteration):
        if iteration < self.warmup_iters:
            return self.start_warmup_value + (self.base_value - self.start_warmup_value) * iteration / self.warmup_iters
        elif iteration >= self.total_iters:
            return self.final_value
        else:
            return self.final_value + 0.5 * (self.base_value - self.final_value) * \
                (1 + math.cos(math.pi * (iteration - self.warmup_iters) / (self.total_iters - self.warmup_iters)))
