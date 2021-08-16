import os
import numpy as np
from cleanfid import fid
import torch
import torch.multiprocessing as mp

from run_metrics import get_vgg_features, make_eval_images


class Evaluator():
    def __init__(self, opt, gan_model):
        self.opt = opt
        self.device = 'cpu' if opt.use_cpu else 'cuda'
        self.gan_model = gan_model

        # load vgg features
        self.real_vgg = get_vgg_features(opt.eval_dir, 2500, opt.eval_batch)

        # load fid stats
        with mp.Pool(1) as p:
            self.fid_stat = p.apply(get_fid_stats, (opt.eval_dir, opt.eval_batch))

        # record the best fid so far
        self.best_fid = float('inf')
        if opt.resume_iter is not None:
            load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "best_FID.npy")
            self.best_fid = float(np.load(load_path))

    def run_metrics(self, iters):
        print("Running metrics and gathering images...")
        cache_folder = f'cache_files/{self.opt.name}'

        print("Gathering images...")
        with torch.no_grad():
            make_eval_images(self.gan_model.netG,
                             cache_folder,
                             2500,
                             self.opt.eval_batch,
                             self.device,
                             to_cpu=False)

        torch.cuda.empty_cache()
        with mp.Pool(1) as p:
            metrics = p.apply(metrics_process, (cache_folder, self.fid_stat, self.real_vgg,))

        best_so_far = False
        if metrics['fid'] < self.best_fid:
            self.best_fid = metrics['fid']
            save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            np.save(save_path + "/best_FID.npy", self.best_fid)
            with open(save_path + "/best_iter.txt", 'w') as f:
                f.write('%d\n' % iters)
            best_so_far = True

        print("Ran metrics successfully...")
        return metrics, best_so_far


def metrics_process(cache_folder, fid_stats, vgg_feats):
    metrics = {}
    print("Evaluating FID...")
    metrics['fid'] = fid.compute_fid(cache_folder+'/image/', num_workers=0, dataset_name=fid_stats, dataset_split="custom")
    torch.cuda.empty_cache()

    print("Evaluating P&R...")
    from eval.precision_recall import metrics as pr
    pr.init_tf()
    # Initialize VGG-16.
    feature_net = pr.initialize_feature_extractor()

    # Calculate VGG-16 features.
    fake_feats = pr.get_features(f'{cache_folder}/image/', feature_net, 2500, 10, num_gpus=1)
    state = pr.knn_precision_recall_features(vgg_feats, fake_feats)
    metrics['precision'] = state['precision'][0]
    metrics['recall'] = state['recall'][0]

    return metrics


def get_fid_stats(eval_dir, eval_batch):
    fid_stat = os.path.basename(eval_dir.rstrip('/')) + '_image'
    if not fid.test_stats_exists(fid_stat, 'clean'):
        fid.make_custom_stats(fid_stat, eval_dir + '/image/', batch_size=eval_batch)

    return fid_stat
