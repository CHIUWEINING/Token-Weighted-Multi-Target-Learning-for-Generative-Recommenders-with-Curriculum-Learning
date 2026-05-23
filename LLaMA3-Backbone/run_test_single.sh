# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
DATA_PATH=../data
# If you want instruction-tuned, use: meta-llama/Llama-3.2-3B-Instruct
BASE_MODEL=meta-llama/Llama-3.2-3B
METHOD=ours
temp=1.0
PQ=""
SEED=42
c=0.00002
CORE7=""

CKPT_PATH=../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/
RESULTS_FILE=../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}_${c}.json

python test.py \
  --ckpt_path $CKPT_PATH \
  --base_model $BASE_MODEL \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --test_batch_size 32 \
  --num_beams 20 \
  --test_prompt_ids 0 \
  --index_file ${PQ}.index.json \
  --dist_eval \
  --output_predictions \
  --group_match all
