python train_gcond_transduct.py \
  --dataset pokec_z \
  --nlayers=3 \
  --sgc=1 \
  --lr_feat=1e-4 \
  --lr_adj=1e-4 \
  --r=0.5 \
  --seed=1 \
  --epoch=600 \
  --save=1 \
  --save_dir saved_runze \
  --full_data \
  --full_data_epoch 2000 \
  --full_data_lr 0.001 \
  --full_data_wd 1e-5 \
  --hidden 128 \
  --no_norm_feat \
  --wandb online \
  "$@"
