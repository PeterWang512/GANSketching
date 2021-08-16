import os
import random
import torch

from training import networks


class GANModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = 'cuda' if not self.opt.use_cpu else 'cpu'
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # netD_image is only created if image regularization is applied
        if isinstance(self.netD, list):
            self.netD_sketch, self.netD_image = self.netD
        else:
            self.netD_sketch = self.netD

        # transform modules to convert generator output to sketches, etc.
        self.tf_real = networks.OutputTransform(opt, process=opt.transform_real, diffaug_policy=opt.diffaug_policy)
        self.tf_fake = networks.OutputTransform(opt, process=opt.transform_fake, diffaug_policy=opt.diffaug_policy)

    # Entry point for all calls involving forward pass of deep networks.
    def forward(self, data, mode):
        real_sketch, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss()
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss, interm_imgs = self.compute_discriminator_loss(real_sketch, real_image)
            return d_loss, interm_imgs
        elif mode == 'discriminator-regularize':
            assert not self.opt.no_d_regularize, "Discriminator shouldn't be regularized with --no_d_regularize applied"
            d_reg_loss = self.compute_discriminator_regularization(real_sketch, real_image)
            return d_reg_loss
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        # select the parameters to optimize for G
        if opt.optim_param_g == 'style':
            G_param_names, G_params = get_param_by_name(self.netG, 'style')
        elif opt.optim_param_g == 'w_shift':
            G_param_names, G_params = get_param_by_name(self.netG, 'w_shift')
        else:
            raise ValueError("--optim_param_g tag should be 'style' or 'w_shift', but get ", opt.optim_param_g)
        self.G_param_names, self.G_params = G_param_names, G_params

        # All of D's parameters will be updatable
        D_params = list(self.netD_sketch.parameters())
        if opt.l_image > 0:
            D_params += list(self.netD_image.parameters())
        self.D_params = D_params

        # if lazy regularization applied, alter the learning rate and beta values
        d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1) if not opt.no_d_regularize else 1.
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr * d_reg_ratio
        G_beta1, D_beta1 = beta1, beta1 ** d_reg_ratio
        G_beta2, D_beta2 = beta2, beta2 ** d_reg_ratio

        # create optimizers based on the selected parameters
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(G_beta1, G_beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(D_beta1, D_beta2))

        return optimizer_G, optimizer_D

    def create_loss_fns(self, opt):
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
        if opt.l_weight > 0:
            self.weight_loss = networks.WeightLoss(self.G_params)
        if not opt.no_d_regularize:
            self.d_regularize = networks.RegularizeD()

    def set_requires_grad(self, g_requires_grad=None, d_requires_grad=None):
        if g_requires_grad is not None:
            networks.set_requires_grad(self.G_params, g_requires_grad)
        if d_requires_grad is not None:
            networks.set_requires_grad(self.D_params, d_requires_grad)

    @torch.no_grad()
    def inference(self, noise, trunc_psi=1.0, mean_latent=None, with_tf=False):
        # If mean_latent is None, make one
        if trunc_psi < 1 and mean_latent is None:
            mean_latent = self.get_mean_latent(n_samples=self.opt.latent_avg_samples)

        img = self.netG([noise], truncation=trunc_psi, truncation_latent=mean_latent)[0]

        # if with_tf is True, output the transformed version (im2sketch) as well
        if with_tf:
            return img, self.tf_fake(img, apply_aug=False)
        return img

    @torch.no_grad()
    def get_mean_latent(self, n_samples=8192):
        return self.netG.mean_latent(n_samples)

    def get_generator(self):
        return self.netG

    def get_discriminator_sketch(self):
        return self.netD_sketch

    def get_discriminator_image(self):
        if hasattr(self, 'netD_image'):
            return self.netD_image
        return None

    def save(self, iters):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        torch.save(self.netG.state_dict(), save_path + "G.pth")
        torch.save(self.netD_sketch.state_dict(), save_path + "D_sketch.pth")
        if self.opt.l_image > 0:
            torch.save(self.netD_image.state_dict(), save_path + "D_image.pth")

    def load(self, iters):
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        state_dict_g = torch.load(load_path + "G.pth", map_location=self.device)
        self.netG.load_state_dict(state_dict_g)

        state_dict_d_sketch = torch.load(load_path + "D_sketch.pth", map_location=self.device)
        self.netD_sketch.load_state_dict(state_dict_d_sketch)
        if self.opt.l_image > 0:
            state_dict_d_image = torch.load(load_path + "D_image.pth", map_location=self.device)
            self.netD_image.load_state_dict(state_dict_d_image)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD_sketch = networks.define_D(opt) if opt.isTrain else None

        if opt.g_pretrained != '':
            weights = torch.load(opt.g_pretrained, map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights, strict=False)

        if netD_sketch is not None and opt.d_pretrained != '' and not opt.dsketch_no_pretrain:
            print("Using pretrained weight for D1...")
            weights = torch.load(opt.d_pretrained, map_location=lambda storage, loc: storage)
            netD_sketch.load_state_dict(weights)

        if opt.l_image > 0:
            assert opt.dataroot_image is not None, "dataset for image regularization needed"
            netD_image = networks.define_D(opt)
            if opt.d_pretrained != '':
                print("Using pretrained weight for D_image...")
                weights = torch.load(opt.d_pretrained, map_location=lambda storage, loc: storage)
                netD_image.load_state_dict(weights)
            netD = [netD_sketch, netD_image]
        else:
            netD = netD_sketch

        return netG, netD

    def print_trainable_params(self):
        print()
        print('-------------- Trainables ---------------')
        print("(G trainable parameters)")
        print('\n'.join(self.G_param_names))
        print('----------------- End -------------------')

    # preprocess the input, such as moving the tensors to GPUs
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        # move to GPU and change data types
        # data['noise'] = data.get('noise', None)
        data['image'] = data.get('image', None)
        if self.use_gpu():
            data['sketch'] = data['sketch'].cuda()
            if data['image'] is not None:
                data['image'] = data['image'].cuda()
        return data['sketch'], data['image']

    def compute_generator_loss(self):
        G_losses = {}
        fake_image = self.generate_fake()

        # applying weight regularization
        if self.opt.l_weight > 0:
            G_losses['l_weight'] = self.opt.l_weight * self.weight_loss(self.G_params)

        # applying sketch loss
        fake_transf = self.tf_fake(fake_image)
        pred_fake = self.netD_sketch(fake_transf)
        G_losses['G_sketch'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)

        # applying image regularization
        if self.opt.l_image > 0:
            pred_fake2 = self.netD_image(fake_image)
            G_losses['G_image'] = self.opt.l_image * \
                self.criterionGAN(pred_fake2, True, for_discriminator=False)

        return G_losses, fake_image.detach()

    def compute_discriminator_loss(self, real_sketch, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake()
            fake_image = fake_image.detach()

        # Transform G's output to sketches to apply sketch loss
        fake_transf = self.tf_fake(fake_image)
        real_transf = self.tf_real(real_sketch)
        pred_fake = self.netD_sketch(fake_transf)
        pred_real = self.netD_sketch(real_transf)

        D_losses['D_fake_sketch'] = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True)
        D_losses['D_real_sketch'] = self.criterionGAN(pred_real, True,
                                            for_discriminator=True)

        # Image regularization
        if self.opt.l_image > 0:
            pred_fake2 = self.netD_image(fake_image)
            pred_real2 = self.netD_image(real_image)
            D_losses['D_fake_image'] = self.opt.l_image * \
                self.criterionGAN(pred_fake2, False, for_discriminator=True)
            D_losses['D_real_image'] = self.opt.l_image * \
                self.criterionGAN(pred_real2, True, for_discriminator=True)

        if self.opt.reduce_visuals:
            interm_imgs = {}
        else:
            interm_imgs = {"real_sketch": real_sketch.detach(),
                        "fake_image": fake_image.detach(),
                        "real_inputD": real_transf.detach(),
                        "fake_inputD": fake_transf.detach()}

        return D_losses, interm_imgs

    def compute_discriminator_regularization(self, real_sketch, real_image):
        D_reg_losses = {}

        # R1 regularization for D_sketch
        real_sketch.requires_grad = True
        pred_real = self.netD_sketch(self.tf_real(real_sketch))
        r1_loss = self.d_regularize(pred_real, real_sketch)
        D_reg_losses['D_r1_loss_sketch'] = self.opt.r1 / 2 * r1_loss * self.opt.d_reg_every

        # R1 regularization for D_image (if image regularization is applied)
        if self.opt.l_image > 0:
            real_image.requires_grad = True
            pred_real2 = self.netD_image(real_image)
            r1_loss2 = self.d_regularize(pred_real2, real_image)
            D_reg_losses['D_r1_loss_image'] = self.opt.l_image * \
                self.opt.r1 / 2 * r1_loss2 * self.opt.d_reg_every

        return D_reg_losses

    def generate_fake(self, batch_size=None, style_mix=True, return_latents=False):
        if batch_size is None:
            batch_size = self.opt.batch

        device = 'cuda' if self.use_gpu() else 'cpu'
        style_mix_prob = self.opt.mixing if style_mix else 0.
        noises = mixing_noise(batch_size, self.opt.z_dim, style_mix_prob, device)

        fake_image, latents = self.netG(noises, return_latents=return_latents)
        if return_latents:
            return fake_image, latents
        return fake_image

    def use_gpu(self):
        return not self.opt.use_cpu


def mixing_noise(batch, latent_dim, prob, device):
    """Generate 1 or 2 set of noises for style mixing."""
    if prob > 0 and random.random() < prob:
        return torch.randn(2, batch, latent_dim, device=device).unbind(0)
    else:
        return [torch.randn(batch, latent_dim, device=device)]


def get_param_by_name(net, tgt_param):
    """Get parameters (and their names) that contain tgt_param in net."""
    name_list, param_list = [], []
    for name, param in net.named_parameters():
        if tgt_param in name:  # target layer
            name_list.append(name)
            param_list.append(param)
    return name_list, param_list
