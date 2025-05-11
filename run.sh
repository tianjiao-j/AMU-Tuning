#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export START_TIME_ALL="`date +%Y_%m_%d-%H_%M_%S`"
echo $START_TIME_ALL

for DATASET_NAME in imagenet sun397 food101 stanford_cars ucf101 caltech101 fgvc eurosat oxford_flowers dtd oxford_pets
do
  for SHOT in 16 8 4 2 1
  do
  python train.py \
      --rand_seed 1 \
      --torch_rand_seed 1 \
      --exp_name amu_tuning_50epochs \
      --clip_backbone "RN50" \
      --augment_epoch 1 \
      --alpha 0.5 \
      --lambda_merge 0.35 \
      --train_epoch 50 \
      --lr 1e-3 \
      --batch_size 8 \
      --shots ${SHOT} \
      --root_path '../Tip-Adapter/data' \
      --dataset ${DATASET_NAME} \
      --moco_ep 100
      #--load_aux_weight \
  done
done
