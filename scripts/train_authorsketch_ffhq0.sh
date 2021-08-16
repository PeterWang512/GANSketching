#!/bin/bash
python train.py \
--name authorsketch_ffhq0_augment --size 1024 --batch 1 \
--dataroot_sketch ./data/sketch/by_author/face0 \
--dataroot_image ./data/image/ffhq --l_image 0.5 \
--g_pretrained ./pretrained/stylegan2-ffhq/netG.pth \
--d_pretrained ./pretrained/stylegan2-ffhq/netD.pth \
--transform_fake down2,toSketch,up2,to3ch --transform_real down2,up2,to3ch \
--reduce_visuals --disable_eval --diffaug_policy translation \
