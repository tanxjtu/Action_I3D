import numpy as np
import numbers
import random
import torch
import cv2
from PIL import Image
import collections


class IMG_resize(object):
    def __init__(self, w, h):
        self.W = w
        self.H = h

    def __call__(self, IMG_clip):
        # list input
        for No, IMG in enumerate(IMG_clip):
            IMG_clip[No] = cv2.resize(IMG, (self.W, self.H),
                                      interpolation=cv2.INTER_LINEAR)
        return IMG_clip

class ToTensor(object): # contain normalize
    def __call__(self, frames):
        frames = np.asarray(frames, dtype=np.float32)
        frames = (frames / 255.) * 2 - 1
        clip = torch.from_numpy(frames.transpose([3, 0, 1, 2])).contiguous()
        return clip
        # return torch.from_numpy(frames.transpose([3, 0, 1, 2])).contiguous()


class Normalize_R3D(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, list_IMG):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        out = []
        for IMG in list_IMG:
            img = torch.from_numpy(IMG)
            IMG_T = img.transpose(0, 1).transpose(0, 2).contiguous().float()
            for t, m, s in zip(IMG_T, self.mean, self.std):
                t.sub_(m).div_(s)
            out.append(IMG_T.unsqueeze(0))
        final = torch.cat(out,0)
        Final = final.transpose(0, 1).contiguous()
        return Final


class CenterCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, imgs):
        Images = np.asarray(imgs)
        t, h, w, c = Images.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return Images[:, i:i + th, j:j + tw, :].astype(np.float32)

class CenterCrop_R3D(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, imgs):
        Num = len(imgs)
        h, w, c = imgs[0].shape
        out = []
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))
        for sg_img in imgs:
            out.append(sg_img[i:i + th, j:j + tw, :].astype(np.float32))
        return out

