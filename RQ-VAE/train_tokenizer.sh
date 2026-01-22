DATASET=movieLens
#Random_Hashing_Musical_Instruments Yelp movieLens
export CUDA_VISIBLE_DEVICES=1 && python ./main.py \
  --dataset $DATASET \
  --data_path ../data/$DATASET/$DATASET.emb-llama-td.npy \
  --ckpt_dir ../checkpoint/rqvae/$DATASET/