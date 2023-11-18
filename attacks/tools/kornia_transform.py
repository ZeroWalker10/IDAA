import torch
from kornia.augmentation import (
    CenterCrop,
    ColorJiggle,
    ColorJitter,
    PadTo,
    RandomAffine,
    RandomBoxBlur,
    RandomBrightness,
    RandomChannelShuffle,
    RandomContrast,
    RandomCrop,
    RandomCutMixV2,
    RandomElasticTransform,
    RandomEqualize,
    RandomErasing,
    RandomFisheye,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomHue,
    RandomInvert,
    RandomJigsaw,
    RandomMixUpV2,
    RandomMosaic,
    RandomMotionBlur,
    RandomPerspective,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomResizedCrop,
    RandomRGBShift,
    RandomRotation,
    RandomSaturation,
    RandomSharpness,
    RandomSolarize,
    RandomThinPlateSpline,
    RandomVerticalFlip,
    Resize
)
import numpy as np

class RandomResize:
    def __init__(self, resize_ratio, p):
        self.resize_ratio = resize_ratio
        self.p = p

    def __call__(self, imgs):
        choices = np.random.rand(len(imgs)) < self.p
        if np.sum(choices) > 0:
            width = imgs.shape[-1]
            resize = int(width * self.resize_ratio)

            rndsize = np.random.randint(min(width, resize), max(width, resize))
            resize_method = Resize([rndsize, rndsize])
            pad_method = PadTo(imgs.shape[-2:], 'constant', 0, keepdim=False)

            imgs[choices] = pad_method(resize_method(imgs[choices]))
        return imgs

class MySelf:
    def __init__(self):
        pass

    def __call__(self, imgs):
        return imgs

class RandomLocalMix:
    def __init__(self, mix_num=1, alpha=0.4, ratio=0.7):
        self.mix_num = mix_num
        self.alpha = alpha
        self.ratio = ratio

    def localmix(self, imgs):
        out_imgs = imgs.clone()
        length = len(imgs)
        sz = imgs.size()
        for _ in range(self.mix_num):
            indexes = torch.randperm(length)
            new_imgs = out_imgs[indexes]
            for i in range(length):
                lam = np.random.beta(self.alpha, self.alpha)
                lam = max(lam, 1 - lam)

                # (1 - lam) * new_imgs[i][:, bbx3:bbx4, bby3:bby4]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(sz, lam)
                bbx3, bby3, bbx4, bby4 = self.rand_bbox(sz, lam, bbx2-bbx1, bby2-bby1)
                out_imgs[i][:, bbx1:bbx2, bby1:bby2] = lam * out_imgs[i][:, bbx1:bbx2, bby1:bby2] + \
                       (1 - lam) * new_imgs[i][:, bbx3:bbx4, bby3:bby4]

        return out_imgs

    def rand_bbox(self, size, lam, ws=None, hs=None):
        W = size[2]
        H = size[3]
        if ws is None or hs is None:
            # cut_rat = np.sqrt(1. - lam)
            cut_rat = self.ratio 
            cut_w = np.int32(W * cut_rat)
            cut_h = np.int32(H * cut_rat)
        else:
            cut_w = ws
            cut_h = hs

        # uniform
        if ws is not None and hs is not None:
            cx = np.random.randint(W - ws)
            cy = np.random.randint(H - hs)

            bbx1 = cx
            bbx2 = cx + ws 
            bby1 = cy
            bby2 = cy + hs
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, imgs):
        return self.localmix(imgs)

