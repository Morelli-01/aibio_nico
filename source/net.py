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
        self.grl = GradientReversalLayer()

    def get_embedding(self, x):
        return self.backbone(x).squeeze()

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        cls = self.cls_head(self.grl(features))
        return projections, cls

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimCLRLoss(object):

    def __call__(self, device: torch.device, data: tuple, net: torch.nn.Module, stream: torch.cuda.Stream = None, lambda_=0.3):
        x_batch, _, metadata = data

        # view for self supervised learning
        transform = v2.Compose([v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
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

        target = torch.cat([metadata[2], metadata[2]]).to(device)

        # cls_loss = 1 / F.cross_entropy(input=cls_digits, target=target)
        cls_loss = F.cross_entropy(input=cls_digits, target=target)
        wandb.log({"unsup loss": unsup_loss.item()})
        wandb.log({"cls loss": cls_loss.item()})
        return unsup_loss + lambda_ * cls_loss

    def info_nce_loss(self, features, device, temperature=1):
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
