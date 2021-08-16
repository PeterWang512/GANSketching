import os
import argparse
from util import util


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--dataroot_sketch", type=str, required=True, help="root to the sketch dataset")
    parser.add_argument("--dataroot_image", type=str, default=None, help="root to the image dataset for image regularization")
    parser.add_argument("--eval_dir", type=str, default=None, help="directory to the evaluation set")
    parser.add_argument("--sketch_channel", type=int, default=1, help="number of channels of sketch inputs, default is monochrome (1)")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoint", help="directory to the checkpoints")
    parser.add_argument("--use_cpu", action="store_true", help="use this flag to operate in cpu mode")

    parser.add_argument("--no_wandb", action="store_true", help="use this flag to disable wandb visualization")
    parser.add_argument("--no_html", action="store_true", help="use this flag to disable html visualization")
    parser.add_argument('--display_winsize', type=int, default=400, help='display window size')
    parser.add_argument("--print_freq", type=int, default=100, help="frequency to print out current logs in stdout")
    parser.add_argument("--display_freq", type=int, default=2500, help="frequency to display visualizations")
    parser.add_argument("--save_freq", type=int, default=2500, help="frequency to save model checkpoints")
    parser.add_argument("--eval_freq", type=int, default=5000, help="frequency to evaluate current results")
    parser.add_argument("--disable_eval", action="store_true", help="use this flag to disable evaluation during training")
    parser.add_argument("--eval_batch", type=int, default=50, help="batch size used to generate images for evaluation")
    parser.add_argument("--reduce_visuals", action="store_true", help="use this flag to reduce amount of visualization (useful in the FFHQ case to reduce memory usage)")
    parser.add_argument("--latent_avg_samples", type=int, default=8192, help="number of samples used to calculate mean latent for truncation")

    parser.add_argument("--resume_iter", type=int, default=None, help="which iteration to resume training, train from scratch if None is given")
    parser.add_argument("--max_iter", type=int, default=75001, help="which iteration to stop training")
    parser.add_argument("--max_epoch", type=int, default=1000000, help="max number of training epoch")

    parser.add_argument("--batch", type=int, default=4, help="batch size used for training")
    parser.add_argument("--size", type=int, default=256, help="image size for StyleGAN2")
    parser.add_argument("--z_dim", type=int, default=512, help="dimensionality of the noise z")
    parser.add_argument("--n_mlp", type=int, default=8, help="number of layers for the style mapping network")
    parser.add_argument("--mixing", type=float, default=0.9, help="style mixing probability used for training")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier for StyleGAN2")

    parser.add_argument("--optim_param_g", type=str, default="style", choices=["style", "w_shift"], help="choose the parameter subset to train: (style) tunes the style mapping network, (w_shift) tunes just a shift to the W space")
    parser.add_argument("--g_pretrained", type=str, default="", help="path to the pre-trained generator")
    parser.add_argument("--d_pretrained", type=str, default="", help="path to the pre-trained discriminator")
    parser.add_argument("--dsketch_no_pretrain", action="store_true", help="use this flag to randomly initialize the sketch discriminator")

    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--lr_mlp", type=float, default=0.01, help="multiplicative factor applied to the learning rate of the style mapping network")

    parser.add_argument("--gan_mode", type=str, default="softplus", help="which gan loss to use? [ls|original|w|hinge|softplus]")
    parser.add_argument("--l_image", type=float, default=0, help="strength of image regularization loss")
    parser.add_argument("--l_weight", type=float, default=0, help="strength of weight regularization loss")
    parser.add_argument("--no_d_regularize", action="store_true", help="use this flag to disable R1 regularization")
    parser.add_argument("--d_reg_every", type=int, default=16, help="frequency to apply R1 regularization")
    parser.add_argument("--r1", type=float, default=10, help="strength of R1 regularzation")

    parser.add_argument("--transform_real", type=str, default='to3ch', help="sequence of operations to transform the real sketches before D")
    parser.add_argument("--transform_fake", type=str, default='toSketch,to3ch', help="sequence of operations to transform the fake images before D")
    parser.add_argument("--photosketch_path", type=str, default='./pretrained/photosketch.pth', help="path to the photosketch pre-trained model")
    parser.add_argument("--diffaug_policy", type=str, default='', help='sequence of operations used for differentiable augmentation')

    opt = parser.parse_args()
    return opt, parser


def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    try:
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    except PermissionError as error:
        print("permission error {}".format(error))
        pass
