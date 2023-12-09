# Pass the dataset location in as the argument.
echo "> Training with dataset location: '$1'"

# Run training script to pre-train with the custom dataset.
python AdaFace/main.py \
    --data_root "data" \
    --train_data_path $1 \
    --val_data_path "faces_real" \
    --prefix "pretrain_$1" \
    --use_wandb \
    --gpus 1 \
    --use_16bit \
    --arch ir_101 \
    --batch_size 64 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.01 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2


# Run training script to train on the real dataset.
python AdaFace/main.py \
    --data_root "data" \
    --start_from_model_statedict "experiments/pretrain_$1_0/last.ckpt" \
    --train_data_path "faces_real" \
    --val_data_path "faces_real" \
    --prefix "real_pretrain_$1" \
    --use_wandb \
    --use_mxrecord \
    --gpus 1 \
    --use_16bit \
    --arch ir_101 \
    --batch_size 64 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.01 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2