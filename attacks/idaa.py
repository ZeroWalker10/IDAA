#!/usr/bin/env python
# coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import kornia
from .attack import Attack
from .tools import kornia_transform
import pdb

class IDAA(Attack):
    def __init__(self, model, aug_library=None, eps=0.07, alpha=1.0, steps=10, mixup_weight=0.3, p=0.0, n=10,
            beta1=0.99, beta2=0.999, mixup_num=3, gamma=0.1, mixup_ratio=0.3):
        super().__init__("IDAA", model)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha 
        self.supported_mode = ["default", "targeted"]

        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-8
        self.mixup_ratio = mixup_ratio

        self.aug_p = 1.0
        self.aug_library = aug_library
        if self.aug_library is None:
            self.aug_library = {
            'geometric': [
                kornia_transform.MySelf(),
                kornia_transform.RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomVerticalFlip(same_on_batch=False, keepdim=False, p=0.6, p_batch=self.aug_p),
                kornia_transform.RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomResize(0.9, p=self.aug_p),
                kornia_transform.RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest",
                    padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p,),
                kornia_transform.RandomErasing(scale=(0.01, 0.04), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomElasticTransform((27, 27), (33, 31), (0.1, 1.0), align_corners=True, padding_mode="reflection", same_on_batch=False, keepdim=False, p=self.aug_p),
                kornia_transform.RandomFisheye(kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([0.9, 1.0]), 
                    same_on_batch=False, keepdim=False, p=self.aug_p),
                ]

            }

        self.nlib = len(self.aug_library['geometric'])
        self.gcandidates = np.arange(self.nlib)
        self.n = n 

        self.mixup_weight = mixup_weight
        self.p = p
        self.rand_local_mix = kornia_transform.RandomLocalMix(mixup_num, mixup_weight, mixup_ratio)

    def forward(self, images, labels, *args, **kwargs):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
            source_labels = kwargs.get('source_labels', None)
        else:
            source_labels = labels


        pred_loss = torch.nn.CrossEntropyLoss() 
        adv_images = images.clone().detach()
        for i in range(len(images)):
            src_image = images[i]
            adv_image = adv_images[i].unsqueeze(0)

            source_label = source_labels[i]
            target_label = target_labels[i]
            pos_labels = torch.LongTensor([int(target_label)] * (self.n + 1)).to(self.device)
            neg_labels = torch.LongTensor([int(source_label)] * (self.n + 1)).to(self.device)

            src_delta_lower_bound = -torch.minimum(src_image.cpu(), torch.Tensor([self.eps])).to(self.device)
            src_delta_upper_bound = torch.minimum(1.0 - src_image.cpu(), torch.Tensor([self.eps])).to(self.device)
            src_delta_bound = src_delta_upper_bound - src_delta_lower_bound

            cur_aug_images = src_image.repeat((self.n+1, 1, 1, 1))
            delta = torch.randn_like(src_image).unsqueeze(0)
            m, v = 0, 0
            for step in range(self.steps):
                deltas = delta.repeat((self.n+1, 1, 1, 1))

                deltas.requires_grad = True
                norm_deltas = src_delta_lower_bound + src_delta_bound * (torch.tanh(deltas.to(self.device)) / 2.0 + 0.5)
                aug_adv_images = cur_aug_images + norm_deltas

                aug_adv_images = torch.cat([
                    self.aug_library['geometric'][k%self.nlib](aug_adv_images[[k]])
                    for k in range(len(aug_adv_images))
                    ])

                aug_adv_images = self.rand_local_mix(aug_adv_images)
                logits = self.model(aug_adv_images)
                if self.targeted: 
                    # targeted
                    pred_cost = 1.0 * pred_loss(logits, pos_labels)
                    ent_cost = -self.gamma * pred_loss(logits, neg_labels)
                    
                    cost = pred_cost + ent_cost

                grad = torch.autograd.grad(cost, deltas, 
                    retain_graph=False, create_graph=False)[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

                grad = grad.mean(0)
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad * grad 
                delta = delta - self.alpha * m / (torch.sqrt(v) + self.epsilon)
            adv_images[i] = src_image.squeeze(0) + (src_delta_lower_bound + src_delta_bound * (torch.tanh(delta.squeeze(0)) / 2.0 + 0.5))
        return adv_images
