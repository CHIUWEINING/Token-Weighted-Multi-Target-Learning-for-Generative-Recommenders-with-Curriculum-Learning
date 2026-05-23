#movieLens Random_Hashing_Musical_Instruments  Random_Hashing_Industrial_and_Scientific Yelp
DATASET=Random_Hashing_Industrial_and_Scientific
METHODS=(origin ours) #
temp=1.0 #0.6
PQ=""
DEVICE=0
# SEED=42
# cs=(0.00005 0.00004 0.00003 0.00002 0.00001)
c=0.000025
SEEDS=(40)
CORE7=""

BASE_MODEL=meta-llama/Llama-3.2-3B
DATA_PATH=../data

# ==== Run Loop ====
for SEED in "${SEEDS[@]}"; do
    echo "=============================="
    echo "Running with SEED = $SEED", " c = $c"
    echo "=============================="
    for METHOD in "${METHODS[@]}"; do
    # for c in "${cs[@]}"; do
        echo "=============================="
        echo "Running with METHOD = $METHOD"
        echo "=============================="

        CKPT_DIR="../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/"
        RESULTS_FILE="../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}_${c}.json"

        export CUDA_VISIBLE_DEVICES=$DEVICE
        export NCCL_P2P_DISABLE="1"
        export NCCL_IB_DISABLE="1"
        export TOKENIZERS_PARALLELISM=false

        # ---- Train ----
        if [ -f "${CKPT_DIR}/adapter_model.safetensors" ] && [ -f "${CKPT_DIR}/adapter_config.json" ]; then
            echo "⏭️  Skip train: found completed checkpoint in ${CKPT_DIR}"
        else
            python lora_finetune.py \
                --base_model $BASE_MODEL \
                --output_dir $CKPT_DIR \
                --dataset $DATASET \
                --data_path $DATA_PATH \
                --per_device_batch_size 16 \
                --learning_rate 1e-4 \
                --epochs 3 \
                --tasks seqrec \
                --train_prompt_sample_num 1 \
                --train_data_sample_num 0 \
                --index_file ${PQ}.index.json \
                --temperature $temp \
                --only_train_response \
                --no_sample_valid \
                --valid_prompt_id 0 \
                --valid_prompt_sample_num 1 \
                --seed $SEED \
                --method $METHOD \
                --c $c \
                --dataloader_num_workers 4 \
                --group_by_length \
                --tf32 \
                --no_gradient_checkpointing
                # --core7 $CORE7
        fi

        # ---- Test ----
        python test.py \
            --ckpt_path $CKPT_DIR \
            --base_model $BASE_MODEL \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --results_file $RESULTS_FILE \
            --test_batch_size 32 \
            --num_beams 10 \
            --test_prompt_ids 0 \
            --index_file ${PQ}.index.json \
            --dist_eval \
            --output_predictions \
            --group_match all
            # --core7 $CORE7 \

        echo "✅ Finished METHOD = $METHOD, SEED = $SEED"
        echo ""
    done
done
