import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torch import nn
import torch, torchvision
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from source.dataset import *
from source.utils import load_yaml, load_device, config_loader
from source.head.trainer_head import Trainer
import copy


class NetWrapper(nn.Module):
    def __init__(self, backbone_net: nn.Module, num_classes):
        super(NetWrapper, self).__init__()
        self.backbone = copy.deepcopy(backbone_net)

        # for p in self.backbone.parameters():
        #     p.requires_grad = False
        # print(f"Backbone weights had been freezed")

        assert hasattr(self.backbone, "embed_dim"), "NetWrapper was unable to parse the embed_dim"
        self.embed_dim = self.backbone.embed_dim
        self.num_classes = num_classes

        self.cls_fc1 = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim * 2)
        self.cls_fc2 = nn.Linear(self.backbone.embed_dim * 2, self.num_classes)

    def forward(self, x):
        features = self.backbone.get_embedding(x).squeeze()
        return self.cls_fc2(F.relu(self.cls_fc1(features)))


def load_backbone(net, config):
    if config["load_backbone"] is not None:
        if "nmorelli" not in config["load_backbone"]:
            import numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(config["load_backbone"], weights_only=False)

            state_dict = checkpoint["student"]
            new_state_dict = {}

            for k in state_dict.keys():
                # if "mlp" in k:
                #     continue
                # if "head" in k:
                #     continue

                new_k = k.removeprefix("module.")
                new_state_dict[new_k] = state_dict[k]
            print(net.load_state_dict(new_state_dict, strict=False))
        else:
            checkpoint = torch.load(config["load_backbone"])
            state_dict = {}
            for k in checkpoint["model_state_dict"].keys():
                # if "backbone" not in k: continue
                new_key = k.removeprefix("module.")
                state_dict[new_key] = checkpoint["model_state_dict"][k]
            print(net.load_state_dict(state_dict, strict=False))
        print(f"Weight found and loaded!!!{config['load_backbone']}")
        return net
    return net


if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)
    dataset = Rxrx1(root_dir=config["dataset_dir"], metadata_path=config["metadata_path"],
                    dataset_norm=config["dataset_norm"])
    net, losser, opt, sched = config_loader(config, dataset)
    net = load_backbone(net, config)
    net = NetWrapper(backbone_net=net, num_classes=dataset.num_classes)

    trainer = Trainer(net=net,
                      device=device,
                      config=config,
                      opt=opt,
                      losser=losser,
                      scheduler=sched,
                      collate=None,
                      dataset=dataset)

    trainer.train()
    exit(0)
