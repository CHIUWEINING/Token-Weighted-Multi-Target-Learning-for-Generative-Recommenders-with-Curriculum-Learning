#Random_Hashing_Musical_Instruments Yelp movieLens
# ========== Config ==========
DATASET=movieLens
METHOD=ours # igd, cft, front_gain, ours, origin, combined
temp=1.0 #0.6
PQ=""
DEVICE=0
SEEDS=(22)

CORE7=""
export CUDA_VISIBLE_DEVICES=$DEVICE
for SEED in "${SEEDS[@]}"; do
  python test.py \
    --ckpt_path ../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}/$DATASET \
    --dataset $DATASET \
    --results_file ../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}.json \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file ${PQ}.index.json \
    --dist_eval \
    --output_predictions \
    --group_match all \
    # --core7 $CORE7 \
done