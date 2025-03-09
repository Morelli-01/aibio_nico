import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from dataset import USDataset
from source.utils import *
from source.backbone.trainer_backbone import *
from torchvision.datasets import ImageFolder
from source.dataset import Rxrx1


def load_backbone_(net, config):
    if config["load_backbone"] is not None:
        checkpoint = torch.load(config["load_backbone"])
        net.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Weight found and loaded!!!{config['load_backbone']}")
        return net
    return net


if __name__ == '__main__':

    config = load_yaml()
    device = load_device(config)

    transform = transforms.Compose([
        transforms.ToImage(),
    ])
    # dataset = ImageFolder(config["dataset_dir"], transform=transform)
    dataset = Rxrx1(root_dir=config["dataset_dir"], metadata_path=config["metadata_path"])
    net, losser, opt, sched = config_loader(config, dataset)
    net = load_backbone_(net, config)

    tr_ = Trainer(net, device, config, opt, losser, collate=None, scheduler=sched)
    tr_.train(dataset)

    exit(0)
