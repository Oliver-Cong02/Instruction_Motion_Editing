torchrun --nnodes 1 --nproc_per_node 4 --master_port=40000 train_t2m.py \
    --pretrained_model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --train_batch_size 4 \
    --num_train_epochs 1000 \
    --checkpointing_steps 10 \
    --dataset_path data/motion_editing/motionfix_val_embeddings.pt \
    --output_dir ./test_output
