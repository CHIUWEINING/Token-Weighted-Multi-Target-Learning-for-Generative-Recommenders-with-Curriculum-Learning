#movieLens Random_Hashing_Musical_Instruments  Random_Hashing_Industrial_and_Scientific Yelp
DATASET=Random_Hashing_Industrial_and_Scientific
METHODS=(ours) #
temp=1.0 #0.6
PQ=""
DEVICE=0
# SEED=42
# cs=(0.00005 0.00004 0.00003 0.00002 0.00001)
c=0.00002
SEEDS=(35 36 37 38 39 40)
CORE7=""
# ==== Run Loop ====
for METHOD in "${METHODS[@]}"; do
    echo "=============================="
    echo "Running with METHOD = $METHOD"
    echo "=============================="
    for SEED in "${SEEDS[@]}"; do
    # for c in "${cs[@]}"; do
        echo "=============================="
        echo "Running with SEED = $SEED", " c = $c"
        echo "=============================="

        export CUDA_VISIBLE_DEVICES=$DEVICE
        export NCCL_P2P_DISABLE="1"
        export NCCL_IB_DISABLE="1"

        # ---- Train ----
        python ./finetune.py \
            --output_dir ../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/ \
            --dataset $DATASET \
            --per_device_batch_size 256 \
            --learning_rate 5e-4 \
            --epochs 200 \
            --index_file ${PQ}.index.json \
            --temperature $temp \
            --seed $SEED \
            --method $METHOD \
            --c $c
            # --core7 $CORE7

        # ---- Test ----
        python test.py \
            --ckpt_path ../checkpoint/transformer${METHOD}${PQ}_seed${SEED}${CORE7}_${c}/$DATASET \
            --dataset $DATASET \
            --results_file ../data/$DATASET/perf${METHOD}${PQ}_seed${SEED}${CORE7}_${c}.json \
            --test_batch_size 32 \
            --num_beams 20 \
            --test_prompt_ids 0 \
            --index_file ${PQ}.index.json \
            --dist_eval \
            --output_predictions \
            --group_match all
            # --core7 $CORE7 \

        echo "✅ Finished SEED = $SEED"
        echo ""
    done
done