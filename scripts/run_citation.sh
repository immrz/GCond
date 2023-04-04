#for lr in 1e-2 1e-3 5e-4 1e-4 5e-5; do
#  for s in 0 1 2 3 4; do
#    python train_gcond_transduct.py \
#      --dataset "$1" \
#      --save=0 \
#      --group_method degree \
#      --full_data \
#      --full_data_epoch 1000 \
#      --nlayers 2 \
#      --full_data_lr $lr \
#      --seed=$s \
#      --wandb online \
#      --wandb_group citation > "stdout/$1_full_data_lr_${lr}_seed_${s}.txt"
#      echo "done $1 $lr $s"
#  done
#done

# for r in 0.25 0.5 1; do
#   for s in 0 1 2 3 4; do
#     echo "seed $s, r $r"
#     python train_gcond_transduct.py \
#       --dataset "$1" \
#       --save=0 \
#       --group_method degree \
#       --r=$r \
#       --seed=$s \
#       --load_exist &> "stdout/$1_condense_r_${r}_seed_${s}.txt"
#   done
# done

for lambda in 0.1 0.5 1; do
  python train_gcond_transduct.py \
    --dataset cora \
    --nlayers=2 \
    --sgc=1 \
    --lr_feat=1e-4 \
    --gpu_id=0 \
    --lr_adj=1e-4 \
    --r=0.5 \
    --seed=1 \
    --epoch=600 \
    --save=0 \
    --demd_lambda ${lambda} \
    --group_method degree \
    --wandb online \
    --wandb_group citation_demd &> "stdout/cora_condense_demd_r_0.5_seed_1_lambda_${lambda}.txt" &
done
