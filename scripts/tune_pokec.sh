#for m in "gcn" "gcn_pokec"
#do
#  for nl in 2 3
#  do
#    for h in 128 256
#    do
#      for d in "pokec_z" "pokec_n"
#      do
#        for r in 0.5 1
#        do
#          ./scripts/run_pokec.sh --inner_model "${m}" --nlayers "${nl}" --hidden "${h}" \
#            --dataset "${d}" --r="${r}" --wandb online --group_method sens
#        done
#      done
#    done
#  done
#done

for nl in 2 3
do
  for h in 128 256
  do
    for d in "pokec_z" "pokec_n"
    do
      ./scripts/run_pokec.sh --nlayers "${nl}" --hidden "${h}" --dataset "${d}" --wandb online \
        --group_method sens --full_data --full_data_lr 0.001 --full_data_wd 1e-5 --full_data_epoch 2000
    done
  done
done
