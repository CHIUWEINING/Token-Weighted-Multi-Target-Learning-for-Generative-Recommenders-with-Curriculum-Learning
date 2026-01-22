#Random_Hashing_Musical_Instruments  Random_Hashing_Industrial_and_Scientific Yelp movieLens
# Random_Hashing_Musical_Instruments origin 42 uncertainty_pace 22 _token_freq(need to revise ckpt path)
# Yelp origin(no seed: need to revise ckpt path) uncertainty 32 _token_freq(need to revise ckpt path)
# movieLens origin 11 uncertainty 22 _token_freq(need to revise ckpt path)
# ========== Config ==========
DATASET=Random_Hashing_Musical_Instruments
METHOD=_token_freq #origin uncertainty
temp=1.0 #0.6
PQ=""
DEVICE=0
SEEDS=("42" "32" "22") #(42 12)
CORE7=""
c=0.00005
MODES=("cold" "warm")
export CUDA_VISIBLE_DEVICES=$DEVICE
for MODE in "${MODES[@]}"; do
  echo "=============================="
  echo "Running EVAL MODE = $MODE"
  echo "=============================="
  for SEED in "${SEEDS[@]}"; do
      echo "=============================="
      echo "Running with SEED = $SEED"
      echo "=============================="
      python test.py \
        --ckpt_path ../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/$DATASET \
        --dataset $DATASET \
        --results_file ../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}_${c}_${MODE}.json \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file ${PQ}.index.json \
        --dist_eval \
        --output_predictions \
        --group_match all \
        --eval_mode $MODE
  done
done