# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
# If you want instruction-tuned, use: meta-llama/Llama-3.2-3B-Instruct
BASE_MODEL=meta-llama/Llama-3.2-3B
DATA_PATH=../data
METHOD=ours
temp=1.0
PQ=""
SEED=42
c=0.000025
CORE7=""

OUTPUT_DIR=../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/

python lora_finetune.py \
  --base_model $BASE_MODEL \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --per_device_batch_size 16 \
  --learning_rate 1e-4 \
  --epochs 4 \
  --tasks seqrec \
  --train_prompt_sample_num 1 \
  --train_data_sample_num 0 \
  --index_file ${PQ}.index.json \
  --wandb_run_name ${METHOD}_${DATASET}_seed${SEED} \
  --only_train_response \
  --no_sample_valid \
  --valid_prompt_id 0 \
  --valid_prompt_sample_num 1 \
  --temperature $temp \
  --method $METHOD \
  --seed $SEED \
  --c $c
