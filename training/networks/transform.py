import torch
import torch.nn as nn
import torch.nn.parallel

from . import pix2pix
from .misc import set_requires_grad
from .diffaug import DiffAugment


class RepeatChannel(nn.Module):
    def __init__(self, repeat):
        super(RepeatChannel, self).__init__()
        self.repeat = repeat

    def forward(self, img):
        return img.repeat(1, self.repeat, 1, 1)


class Downsample(nn.Module):
    def __init__(self, n_iter):
        super(Downsample, self).__init__()
        self.n_iter = n_iter

    def forward(self, img):
        for _ in range(self.n_iter):
            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bicubic')
        return img


class Upsample(nn.Module):
    def __init__(self, n_iter):
        super(Upsample, self).__init__()
        self.n_iter = n_iter

    def forward(self, img):
        for _ in range(self.n_iter):
            img = nn.functional.interpolate(img, scale_factor=2.0, mode='bicubic')
        return img


class OutputTransform(nn.Module):
    def __init__(self, opt, process='', diffaug_policy=''):
        super(OutputTransform, self).__init__()
        self.opt = opt
        if diffaug_policy == '':
            self.augment = None
        else:
            self.augment = DiffAugment(policy=diffaug_policy)

        transforms = []
        process = process.split(',')
        for p in process:
            if p.startswith('up'):
                n_iter = int(p.replace('up', ''))
                transforms.append(Upsample(n_iter))
            elif p.startswith('down'):
                n_iter = int(p.replace('down', ''))
                transforms.append(Downsample(n_iter))
            elif p == 'to3ch':
                transforms.append(RepeatChannel(3))
            elif p == 'toSketch':
                sketch = self.setup_sketch(opt)
                transforms.append(sketch)
            else:
                ValueError("Transforms contains unrecognizable key: %s" % p)
        self.transforms = nn.Sequential(*transforms)

    def setup_sketch(self, opt):
        sketch = pix2pix.ResnetGenerator(3, 1, n_blocks=9, use_dropout=False)

        state_dict = torch.load(opt.photosketch_path, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        sketch.load_state_dict(state_dict)
        sketch.train()
        set_requires_grad(sketch.parameters(), False)
        return sketch

    def forward(self, img, apply_aug=True):
        img = self.transforms(img)
        if apply_aug and self.augment is not None:
            img = self.augment(img)
        return img
