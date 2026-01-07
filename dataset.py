import argparse
import json
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import joblib


import torch
from torch.utils.data import Dataset

class MotionFixDataset(Dataset):
    def __init__(self, dataset_path, max_motion_length=196, max_ctxt_len=128, vtxt_dim=768):
        self.data_dict = torch.load(dataset_path, map_location='cpu')
        
        self.keys = list(self.data_dict.keys())
        self.max_motion_length = max_motion_length
        self.max_ctxt_len = max_ctxt_len
        self.vtxt_dim = vtxt_dim

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item_key = self.keys[idx]
        item_data = self.data_dict[item_key]
        
        text_input = item_data.get('rewritten_text', "")
        motion_latents = item_data['tgt_latent'].float()
        breakpoint()
        original_motion_length = motion_latents.shape[0]
        motion_lengths = torch.tensor([original_motion_length])

        text_embedding = item_data['text_embedding']
        vtxt_input = text_embedding['vtxt_input'].float()
        ctxt_input_raw = text_embedding['ctxt_input'].float()
        ctxt_length = text_embedding['ctxt_length']
        ctxt_mask_temporal = text_embedding['ctxt_mask_temporal'].float()
        
        # 获取 embedding 的真实长度
        valid_ctxt_len = ctxt_input_raw.shape[1]
        
        if valid_ctxt_len < self.max_ctxt_len:
            padding = torch.zeros(self.max_ctxt_len - valid_ctxt_len, ctxt_input_raw.shape[1])
            ctxt_input = torch.cat([ctxt_input_raw, padding], dim=0)
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal, torch.zeros(1, self.max_ctxt_len - valid_ctxt_len)], dim=1)
        else:
            raise ValueError(f"Context length {valid_ctxt_len} exceeds max_ctxt_len {self.max_ctxt_len}")
            ctxt_input = ctxt_input_raw[:self.max_ctxt_len]

        # -----------------------------------------------------------
        # 5. 映射 vtxt_input
        # 源数据中没有 'vtxt_input'，为了保持结构一致，创建一个全0的占位符或随机张量
        # DumpDataset 中形状是 (1, vtxt_dim)
        vtxt_input = torch.zeros(1, self.vtxt_dim) 

        return {
            "text_inputs": text_input,            # str
            "motion_latents": motion_latents,     # Tensor
            "motion_lengths": motion_lengths,     # Tensor
            "ctxt_input": ctxt_input,             # Tensor (padded)
            "vtxt_input": vtxt_input,             # Tensor (placeholder)
            "text_ctxt_raw_length": text_ctxt_raw_length # Tensor
        }
    
    def collate_fn(self, batch):
        # 保持与 DumpMotionDataset 一致的逻辑
        # 注意：如果 motion_latents 长度不一致，这里直接 stack 会报错。
        # 通常需要 pad_sequence。为了严格仿照你的代码，这里保留 stack，
        # 但在 __getitem__ 中你可能需要确保 motion 也是固定长度，或者在这里修改为 padding。
        
        text_inputs = [item["text_inputs"] for item in batch]
        
        # 这里的 stack 假设所有 motion_latents 长度一致。
        # 如果源数据长度不一，建议使用 torch.nn.utils.rnn.pad_sequence
        try:
            motion_latents = torch.stack([item["motion_latents"] for item in batch])
        except RuntimeError:
            # 如果长度不一致，自动退回到 padding 模式 (更健壮)
            from torch.nn.utils.rnn import pad_sequence
            motion_latents = pad_sequence([item["motion_latents"] for item in batch], batch_first=True)

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

    for idx, data in enumerate(dataset):
        pass
    
    
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
