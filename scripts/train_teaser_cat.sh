#!/bin/bash
python train.py \
--name teaser_cat_augment --batch 4 \
--dataroot_sketch ./data/sketch/by_author/cat \
--dataroot_image ./data/image/cat --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-cat/netG.pth \
--d_pretrained ./pretrained/stylegan2-cat/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation \
