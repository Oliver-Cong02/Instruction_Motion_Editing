torchrun --nnodes 1 --nproc_per_node 8 --master_port=40000 train_t2m.py \
    --config_path ckpts/tencent/HY-Motion-1.0-Lite/config.yml \
    --train_batch_size 4 \
    --num_train_epochs 1000 \
    --dataset_path ./motionfix_test_embeddings.pt \
    --output_dir ./test_output
