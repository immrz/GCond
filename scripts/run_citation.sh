#!/bin/bash

MAX_JOBS=6
OUTPUT_DIR="tmp"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p $OUTPUT_DIR
fi

if [[ -z $1 ]]; then
  echo "Specify dataset: cora or citeseer"
  exit 1
fi

if [[ -z $2 ]]; then
  echo "Specify what you wanna do: original or demd"
  exit 1
fi

DATASET="$1"
METHOD="$2"

run_one () {
  python -u train_gcond_transduct.py \
    --dataset "$DATASET" \
    --nlayers=2 \
    --sgc=1 \
    --lr_feat=1e-4 \
    --gpu_id=0 \
    --lr_adj=1e-4 \
    --r="${redu}" \
    --seed="$seed" \
    --epoch=600 \
    --save=0 \
    --demd_lambda "$lambda" \
    --demd_bins "$bins" \
    --group_method degree \
    --groupby_degree_thres "$thres" \
    --wandb offline \
    --wandb_group citation_demd_2
}

run_original () {
  python train_gcond_transduct.py --dataset "$DATASET" --nlayers=2 --sgc=1 --lr_feat=1e-4 --lr_adj=1e-4 \
    --r="$redu" --seed="$seed" --epoch=600 --save=1 --save_dir "saved_runze/$DATASET"
}

case "$METHOD" in
  "demd")
    # run condensation with demd
    for redu in 0.25 0.5; do
      for thres in 3 4.7 6.5; do
        for bins in 10 20 30; do
          for lambda in 0.001 0.01 0.05 0.1 0.5 1 2; do
            for seed in 0 1 2 3 4; do
              run_one &> "${OUTPUT_DIR}/${DATASET}__r_${redu}__thres_${thres}__bins_${bins}__lambda_${lambda}__seed_${seed}.txt" &
              if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
                wait -n
              fi
            done
          done
        done
      done
    done

    wait
    ;;

  "original")
    # run condensation with original method, and also full data
    for redu in 0.25 0.5; do
      for seed in 0 1 2 3 4; do
        run_original &> "${OUTPUT_DIR}/${DATASET}__r_${redu}__seed_${seed}.txt" &
        if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
          wait -n
        fi
      done
    done
    wait

    for seed in 0 1 2 3 4; do
      python train_gcond_transduct.py --dataset "$DATASET" --nlayers=2 --seed="$seed" --full_data --full_data_epoch 1000 --full_data_lr 5e-4 \
        --save=1 --save_dir "saved_runze/$DATASET" &> "${OUTPUT_DIR}/${DATASET}__full_data__seed_${seed}.txt" &
    done
    wait
    ;;

  *)
    echo "Specify what you wanna do: original or demd"
    exit 1
    ;;
esac
