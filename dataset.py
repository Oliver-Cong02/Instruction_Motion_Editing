import argparse
import json
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import joblib

from hymotion.pipeline.motion_diffusion import length_to_mask


import torch
from torch.utils.data import Dataset

class MotionFixDataset(Dataset):
    def __init__(self, dataset_path, max_x_len=360):
        self.data_dict = torch.load(dataset_path, map_location='cpu')
        
        self.keys = list(self.data_dict.keys())
        self.max_x_len = max_x_len

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item_key = self.keys[idx]
        item_data = self.data_dict[item_key]
        
        text_input = item_data.get('rewritten_text', "")
        motion_latents = item_data['tgt_latent'].float()

        text_embedding = item_data['text_embedding']
        vtxt_input = text_embedding['vtxt_input'].float()
        ctxt_input = text_embedding['ctxt_input'].float()
        ctxt_length = text_embedding['ctxt_length']

        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])

        curr_len = motion_latents.shape[0]
        dim = motion_latents.shape[1]
        if curr_len < self.max_x_len:
            pad_size = self.max_x_len - curr_len
            padding = torch.zeros((pad_size, dim), dtype=motion_latents.dtype)
            motion_latents = torch.cat([motion_latents, padding], dim=0)
            x_length = torch.LongTensor([curr_len])
        else:
            raise ValueError(f"Motion length {curr_len} exceeds max_x_len {self.max_x_len}")
        
        x_mask_temporal = length_to_mask(x_length, self.max_x_len)

        return {
            "text_inputs": text_input,           
            "x_latents": motion_latents, # (1, 120, 201)
            "x_length": x_length, # (1,)
            "x_mask_temporal": x_mask_temporal.squeeze(0), # (1, 120)
            "ctxt_input": ctxt_input.squeeze(0),  # (1, 128, 4096)   
            "vtxt_input": vtxt_input.squeeze(0), # (1, 1, 768)
            "ctxt_mask_temporal": ctxt_mask_temporal.squeeze(0), # (1, 128)
        }
    
    def collate_fn(self, batch):
        text_inputs = [item["text_inputs"] for item in batch]
        x_latents = torch.stack([item["x_latents"] for item in batch])
        x_length = torch.cat([item["x_length"] for item in batch])
        x_mask_temporal = torch.stack([item["x_mask_temporal"] for item in batch])
        
        ctxt_input = torch.stack([item["ctxt_input"] for item in batch])
        vtxt_input = torch.stack([item["vtxt_input"] for item in batch])
        ctxt_mask_temporal = torch.stack([item["ctxt_mask_temporal"] for item in batch])
        
        return {
            "text_inputs": text_inputs,
            "x_latents": x_latents,
            "x_length": x_length,
            "x_mask_temporal": x_mask_temporal,
            "ctxt_input": ctxt_input,
            "vtxt_input": vtxt_input,
            "ctxt_mask_temporal": ctxt_mask_temporal,
        }


class DumpMotionDataset(Dataset):
    def __init__(self, num_samples=100000, motion_length=360, motion_dim=201, 
                 ctxt_seq_len=128, ctxt_dim=4096, vtxt_dim=768):
        self.num_samples = num_samples
        self.motion_length = motion_length
        self.motion_dim = motion_dim
        self.ctxt_seq_len = ctxt_seq_len
        self.ctxt_dim = ctxt_dim
        self.vtxt_dim = vtxt_dim
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机的motion_latents: (motion_length, motion_dim)
        motion_latents = torch.randn(self.motion_length, self.motion_dim)
        
        # 生成随机的ctxt_input: (ctxt_seq_len, ctxt_dim)
        ctxt_input = torch.randn(self.ctxt_seq_len, self.ctxt_dim)
        
        # 生成随机的vtxt_input: (1, vtxt_dim)
        vtxt_input = torch.randn(1, self.vtxt_dim)
        
        # 生成文本输入（用于文本编码器）
        text_input = "A person is walking."
        
        # 生成motion_lengths（实际运动长度，这里我们使用完整长度）
        motion_lengths = torch.tensor([self.motion_length])
        
        return {
            "text_inputs": text_input,
            "motion_latents": motion_latents,
            "motion_lengths": motion_lengths,
            "ctxt_input": ctxt_input,
            "vtxt_input": vtxt_input,
            "text_ctxt_raw_length": torch.tensor([10])
        }
    
    def collate_fn(self, batch):
        # 批量处理数据
        text_inputs = [item["text_inputs"] for item in batch]
        motion_latents = torch.stack([item["motion_latents"] for item in batch])
        motion_lengths = torch.cat([item["motion_lengths"] for item in batch])
        ctxt_input = torch.stack([item["ctxt_input"] for item in batch])
        vtxt_input = torch.stack([item["vtxt_input"] for item in batch])
        ctxt_length = torch.cat([item["text_ctxt_raw_length"] for item in batch])
        
        return {
            "text_inputs": text_inputs,
            "motion_latents": motion_latents,
            "motion_lengths": motion_lengths,
            "ctxt_input": ctxt_input,
            "vtxt_input": vtxt_input,
            "text_ctxt_raw_length": ctxt_length
        }

# 创建一个简单的测试函数
def test_dataloader():
    dataset = DumpMotionDataset(
        num_samples=100,
        motion_length=360,
        motion_dim=201,
        ctxt_seq_len=128,
        ctxt_dim=4096,
        vtxt_dim=768
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    # 测试数据加载
    print("Testing dataloader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  text_inputs: list of {len(batch['text_inputs'])} strings")
        print(f"  motion_latents.shape: {batch['motion_latents'].shape}")
        print(f"  motion_lengths.shape: {batch['motion_lengths'].shape}")
        print(f"  ctxt_input.shape: {batch['ctxt_input'].shape}")
        print(f"  vtxt_input.shape: {batch['vtxt_input'].shape}")
        
        # 检查形状是否符合要求
        assert batch['motion_latents'].shape[1:] == (360, 201), f"motion_latents shape mismatch: {batch['motion_latents'].shape}"
        assert batch['ctxt_input'].shape[1:] == (128, 4096), f"ctxt_input shape mismatch: {batch['ctxt_input'].shape}"
        assert batch['vtxt_input'].shape[1:] == (1, 768), f"vtxt_input shape mismatch: {batch['vtxt_input'].shape}"
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("Dataloader test passed!")

if __name__ == "__main__":
    # 测试数据加载器
    # test_dataloader()
    # dataset_dict_raw = joblib.load("motionfix_test_embeddings.pth.tar")
    # dataset = torch.load("motionfix_test_embeddings.pt")


    dataset_path = "/opt/tiger/VIVA/Instruction_Motion_Editing/motionfix_test_embeddings.pt"
    dataset = MotionFixDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)

    for idx, data in enumerate(dataloader):
        print(f"Batch {idx + 1}:")
        print(f"  text_inputs: {data['text_inputs']}")
        print(f"  motion_latents.shape: {data['x_latents'].shape}")
        print(f"  motion_lengths.shape: {data['x_length'].shape}")
        print(f"  ctxt_input.shape: {data['ctxt_input'].shape}")
        print(f"  vtxt_input.shape: {data['vtxt_input'].shape}")
        print(f"  x_mask_temporal.shape: {data['x_mask_temporal'].shape}")
        print(f"  ctxt_mask_temporal.shape: {data['ctxt_mask_temporal'].shape}")
    
    
    print("\n=== 使用说明 ===")
    # print("1. 运行数据加载器测试:")
    # print("   python test_dump_dataset.py")
    
    # print("\n2. 使用dump dataset运行训练脚本:")
    # print("   python -m torch.distributed.launch --nproc_per_node=1 train_t2m.py \")
    # print("   --config_path test_config.yaml \")
    # print("   --dataset_module test_dump_dataset.DumpMotionDataset \")
    # print("   --dataset_args '{\"num_samples\": 1000}' \")
    # print("   --train_batch_size 4 \")
    # print("   --max_train_steps 100 \")
    # print("   --output_dir ./test_output")
