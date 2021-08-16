import time
import torch
import torch.multiprocessing as mp

from options import get_opt, print_options
from eval import Evaluator
from util.visualizer import Visualizer
from training.gan_trainer import GANTrainer
from training.dataset import create_dataloader, yield_data


def training_loop():
    torch.backends.cudnn.benchmark = True

    opt, parser = get_opt()
    opt.isTrain = True

    # needs to switch to spawn mode to be compatible with evaluation
    if not opt.disable_eval:
        mp.set_start_method('spawn')

    # dataloader for user sketches
    dataloader_sketch, sampler_sketch = create_dataloader(opt.dataroot_sketch,
                                                          opt.size,
                                                          opt.batch,
                                                          opt.sketch_channel)
    # dataloader for image regularization
    if opt.dataroot_image is not None:
        dataloader_image, sampler_image = create_dataloader(opt.dataroot_image,
                                                            opt.size,
                                                            opt.batch)
        data_yield_image = yield_data(dataloader_image, sampler_image)

    trainer = GANTrainer(opt)

    print_options(parser, opt)
    trainer.gan_model.print_trainable_params()
    if not opt.disable_eval:
        evaluator = Evaluator(opt, trainer.get_gan_model())
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # the total number of training iterations
    if opt.resume_iter is None:
        total_iters = 0
    else:
        total_iters = opt.resume_iter

    optimize_time = 0.1
    for epoch in range(opt.max_epoch):
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data_sketch in enumerate(dataloader_sketch):  # inner loop within one epoch
            if total_iters >= opt.max_iter:
                return

            # makes dictionary to store all inputs
            data = {}
            data['sketch'] = data_sketch
            if opt.dataroot_image is not None:
                data_image = next(data_yield_image)
                data['image'] = data_image

            # timer for data loading per iteration
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # timer for optimization per iteration
            optimize_start_time = time.time()
            trainer.train_one_step(data, total_iters)
            optimize_time = (time.time() - optimize_start_time) * 0.005 + 0.995 * optimize_time

            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, total_iters, losses, optimize_time, t_data)
                visualizer.plot_current_errors(losses, total_iters)

            # display images on wandb and save images to a HTML file
            if total_iters % opt.display_freq == 0:
                visuals = trainer.get_visuals()
                visualizer.display_current_results(visuals, epoch, total_iters)

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                trainer.save(total_iters)

            # evaluate the latest model
            if not opt.disable_eval and total_iters % opt.eval_freq == 0:
                metrics_start_time = time.time()
                metrics, best_so_far = evaluator.run_metrics(total_iters)
                metrics_time = time.time() - metrics_start_time

                visualizer.print_current_metrics(epoch, total_iters, metrics, metrics_time)
                visualizer.plot_current_errors(metrics, total_iters)

            total_iters += 1
            epoch_iter += 1
            iter_data_time = time.time()


if __name__ == "__main__":
    training_loop()
    print('Training was successfully finished.')
