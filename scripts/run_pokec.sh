python train_gcond_transduct.py \
  --dataset pokec_z \
  --nlayers=3 \
  --sgc=1 \
  --lr_feat=1e-4 \
  --lr_adj=1e-4 \
  --r=0.5 \
  --seed=1 \
  --epochs=1000 \
  --save=1 \
  --save_dir saved_runze \
  --hidden 128 \
  --no_norm_feat \
  --wandb disabled \
  --group_method sens \
  "$@"
