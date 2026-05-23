#Random_Hashing_Musical_Instruments Yelp movieLens Instruments
# ========== Config ==========
DATASET=Instruments
METHOD=ours # igd, cft, front_gain, ours, origin, combine
temp=1.0 #0.6
PQ=""
DEVICE=0
SEEDS=(22)
CORE7=""
c=0.00002
# If you want instruction-tuned, use: meta-llama/Llama-3.2-3B-Instruct
BASE_MODEL=meta-llama/Llama-3.2-3B
DATA_PATH=../data

export CUDA_VISIBLE_DEVICES=$DEVICE
for SEED in "${SEEDS[@]}"; do
  python test.py \
    --ckpt_path ../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/ \
    --base_model $BASE_MODEL \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file ../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}_${c}.json \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file ${PQ}.index.json \
    --dist_eval \
    --output_predictions \
    --group_match all \
    # --core7 $CORE7

done
