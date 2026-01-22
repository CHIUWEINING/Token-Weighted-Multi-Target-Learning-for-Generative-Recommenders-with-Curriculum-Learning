DATASET=movieLens
#Random_Hashing_Musical_Instruments Beauty Yelp Random_Hashing_Industrial_and_Scientific movieLens
export CUDA_VISIBLE_DEVICES=2 && python ./generate_indices.py\
    --dataset $DATASET \
    --data_path  ../data/ \
    --epoch 10000 \
    --root_path ../checkpoint/rqvae/$DATASET/llama/ \
    --checkpoint best_collision_model.pth
    # --alpha_ 0.8