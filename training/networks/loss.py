import torch
from torch import nn, autograd
from torch.nn import functional as F


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        all_gan_modes = ['ls', 'original', 'w', 'hinge', 'softplus']
        if gan_mode not in all_gan_modes:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss

        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)

        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss

        elif self.gan_mode == 'softplus':
            if for_discriminator:
                if target_is_real:
                    loss = F.softplus(-input).mean()
                else:
                    loss = F.softplus(input).mean()
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = F.softplus(-input).mean()
            return loss

        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def forward(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class RegularizeD(nn.Module):
    def forward(self, real_pred, real_img):
        # in case of patchGAN, take average of per-pixel predictions, and sum over batches
        outputs = real_pred.reshape(real_pred.shape[0], -1).mean(1).sum()
        grad_real, = autograd.grad(
            outputs=outputs, inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


class WeightLoss(nn.Module):
    def __init__(self, params):
        super(WeightLoss, self).__init__()
        self.ref_weights = [
            torch.tensor(p, requires_grad=False, device=p.device) for p in params
        ]

    def forward(self, params):
        losses = []
        for i in range(len(params)):
            losses.append((params[i] - self.ref_weights[i]).abs().mean())
        loss = sum(losses) / len(losses)
        return loss
