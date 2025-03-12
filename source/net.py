import os, sys

from cv2 import norm
from sympy import N
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import wandb
from typing import Callable
from torch import nn
from torchvision import models
import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from vision_transformer import *
from PIL import Image
from source import utils
import numpy as np
import torch.distributed as dist


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class SimCLR(nn.Module):
    def __init__(self, get_embeddings=False, backbone="resnet50", num_classes=4):
        super(SimCLR, self).__init__()
        self.get_embeddings = get_embeddings
        self.backbone_name = backbone
        self.num_classes = num_classes
        if "resnet" in self.backbone_name:
            if self.backbone_name == "resnet50":
                self.backbone = models.resnet50(weights="DEFAULT")
            elif self.backbone_name == "resnet101":
                self.backbone = models.resnet101(weights="DEFAULT")
            elif self.backbone_name == "resnet152":
                self.backbone = models.resnet152(weights="DEFAULT")
            self.backbone.fc = nn.Identity()  # fully-connected removed
            self.embed_dim = list(self.backbone.children())[-3][-1].bn3.num_features
        elif "vit" in self.backbone_name:
            if self.backbone_name == "vit_base":
                self.backbone = vit_base()
            elif self.backbone_name == "vit_tiny":
                self.backbone = vit_tiny()
            elif self.backbone_name == "vit_small":
                self.backbone = vit_small()
            elif self.backbone_name == "vit_large":
                self.backbone = vit_large()
            self.embed_dim = self.backbone.embed_dim

        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2, bias=False),
            nn.BatchNorm1d(self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 2, 256, bias=False),
            nn.BatchNorm1d(256)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.num_classes),
        )
        # self.grl = GradientReversalLayer()

    def get_embedding(self, x):
        return self.backbone(x).squeeze()

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        cls = self.cls_head(features)
        return projections, cls

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimCLRLoss(object):

    def __call__(self, device: torch.device, data: tuple, net: torch.nn.Module, stream: torch.cuda.Stream = None, lambda_=0.1):
        x_batch, _, metadata = data

        # view for self supervised learning
        transform = v2.Compose([v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                v2.RandomHorizontalFlip(),
                                v2.RandomVerticalFlip(),
                                v2.ColorJitter(brightness=0.2, contrast=0.2),
                                v2.GaussianBlur(kernel_size=3),
                                v2.ToImage(), v2.ToDtype(torch.float, scale=True)]
                               )
        if stream is not None:
            with torch.cuda.stream(stream):
                xi = transform(x_batch).to(device, non_blocking=True)
                xj = transform(x_batch).to(device, non_blocking=True)

        else:
            xi = transform(x_batch).to(device, non_blocking=True)
            xj = transform(x_batch).to(device, non_blocking=True)

        block = torch.cat([xi, xj], dim=0)
        out_feat, cls_digits = net.forward(block.to(torch.float))
        # unsup_loss = self.compute_loss(out_feat, device)
        unsup_loss = self.info_nce_loss(out_feat, device)  # faster then the above one

        # target = torch.cat([metadata[self.cell_type_idx], metadata[self.cell_type_idx]]).squeeze().to(device)

        # cls_loss = F.cross_entropy(input=cls_digits, target=target)
        # wandb.log({"unsup loss": unsup_loss.item()})
        # wandb.log({"cls loss": cls_loss.item()})
        return unsup_loss

    def info_nce_loss(self, features, device, temperature=0.5):
        """
        Implements Noise Contrastive Estimation loss as explained in the simCLR paper.
        Actual code is taken from here https://github.com/sthalles/SimCLR/blob/master/simclr.py
        Args:
            - features: torch tensor of shape (2*N, D) where N is the batch size.
                The first N samples are the original views, while the last
                N are the modified views.
            - device: torch device
            - temperature: float
        """
        n_views = 2
        assert features.shape[0] % n_views == 0  # make sure shapes are correct
        batch_size = features.shape[0] // n_views

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            n_views * batch_size, n_views * batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return F.cross_entropy(logits, labels)

    def nt_xent(self, i, j, cos_sim):
        loss = cos_sim[i, j]
        loss = loss / cos_sim[i, :].sum()
        return -loss.log()

    def compute_loss(self, features: torch.tensor, device: torch.device, temperature=1):
        loss = 0
        norms = features.norm(dim=-1, keepdim=True)
        cos_sim = (features @ features.T) / (norms @ norms.T)
        cos_sim = (cos_sim / temperature).exp()
        cos_sim = cos_sim * (1.0 - torch.eye(features.shape[0], device=device))
        for i in range(features.shape[0] // 2):
            loss += self.nt_xent(i, i + (features.shape[0] // 2), cos_sim)
            loss += self.nt_xent(i + (features.shape[0] // 2), i, cos_sim)

        loss /= features.shape[0]
        return loss

    def set_experiment_idx(self, idx):
        self.experiment_idx = idx

    def set_cell_type_idx(self, idx):
        self.cell_type_idx = idx


class ClassificationLoss(object):
    def __init__(self, device):
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self, data, net, stream: torch.cuda.Stream = None):
        x_batch, y_batch, metadata = data
        if stream is not None:
            with torch.cuda.stream(stream):
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

        pred = net(x_batch)
        loss = self.loss_func(pred, y_batch)
        return loss


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8):
        flip_and_color_jitter = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            v2.RandomGrayscale(p=0.2),
        ])

        def gaussian_blur(p): return v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=p
        )
        normalize = v2.Compose([
            v2.Identity()
            # v2.ToTensor(),
            # v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = v2.Compose([
            v2.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            gaussian_blur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = v2.Compose([
            v2.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            gaussian_blur(0.1),
            v2.RandomSolarize(0.5, 0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = v2.Compose([
            v2.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            gaussian_blur(0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)
