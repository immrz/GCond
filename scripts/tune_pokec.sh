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

#for nl in 2 3
#do
#  for h in 128 256
#  do
#    for d in "pokec_z" "pokec_n"
#    do
#      ./scripts/run_pokec.sh --nlayers "${nl}" --hidden "${h}" --dataset "${d}" --wandb online \
#        --group_method sens --full_data --full_data_lr 0.001 --full_data_wd 1e-5 --full_data_epoch 2000 --inner_model gcn_pokec
#    done
#  done
#done

#for s in 1 2 10 42 100 786
#do
#  for ln in 1000 1500 2000 3000 4000
#  do
#    ./scripts/run_pokec.sh --nlayers 3 --hidden 256 --dataset pokec_z --wandb online --wandb_group try_seed \
#      --group_method sens --r 500 --seed "${s}" --label_number "${ln}"
#  done
#done

for h in 128 256 512
do
  for nl in 2 3 4
  do
    for sgc in 1 2
    do
      for r in 250 500 800
      do
        for lf in 1e-3
        do
          for s in 2 100 786
          do
            ./scripts/run_pokec.sh --nlayers "${nl}" --hidden "${h}" --sgc "${sgc}" --r "${r}" --seed "${s}" \
              --inner_hidden 128 --inner_nlayers 2 --lr_feat "${lf}" \
              --dataset pokec_z --wandb online --wandb_group tune_pokecz \
              --group_method sens --label_number 2000 --epochs 1600 --one_step --save 0
          done
        done
      done
    done
  done
done
