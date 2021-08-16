#!/bin/bash
python train.py \
--name quickdraw_single_horse0_augment --batch 1 \
--dataroot_sketch ./data/sketch/quickdraw/single_sketch/horse0 \
--dataroot_image ./data/image/horse --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation \
