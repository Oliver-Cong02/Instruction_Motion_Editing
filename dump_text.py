import os
import time
import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import joblib
from hymotion.utils.t2m_runtime import T2MRuntime

def process_single_embedding(
    runtime: T2MRuntime,
    key: str,
    text: str,
    disable_rewrite: bool = False,
    disable_duration_est: bool = False,
) -> Dict[str, Any]:
    """
    处理单个样本的文本，获取 Embedding。
    """
    # print(f">>> Processing key: {key} | text: {text}")

    # 1. LLM 重写与时长预估逻辑 (保留原逻辑，如果不需要可以关掉)
    call_llm = not disable_rewrite or not disable_duration_est
    
    if not call_llm:
        rewritten_text = text
    else:
        # 注意：如果你的 generate_text_embedding 不需要 duration，
        # 这里仅利用 LLM 的 rewrite 功能优化文本
        predicted_duration, rewritten_text = runtime.rewrite_text_and_infer_time(text=text)
        if disable_rewrite:
            rewritten_text = text
            
    # 2. 调用核心 Embedding 生成函数 (你需要在 Runtime 中实现此方法)
    # 假设返回的是 Tensor 或 Numpy Array
    embedding = runtime.generate_text_embedding(text=rewritten_text)

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu()

    return {
        "key": key,
        "original_text": text,
        "rewritten_text": rewritten_text,
        "text_embedding": embedding
    }


def run_parallel_embedding_tasks(
    runtime: T2MRuntime,
    dataset_dict: Dict[str, Dict],
    disable_rewrite: bool = False,
    disable_duration_est: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    并行处理数据集中的文本提取 Embedding
    """
    results_map = {} # 用于存储 {key: embedding}
    
    total_tasks = len(dataset_dict)
    print(f">>> Start processing {total_tasks} items...")

    if max_workers is None:
        # 如果是计算密集型（LLM推理），worker数不宜过多；如果是IO密集型，可以多一点
        max_workers = max(1, len(runtime.device_ids) if runtime.device_ids else 1) * 2

    # 提交任务
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # 将字典转换为任务列表
        futures = {}
        for key, data in dataset_dict.items():
            input_text = data.get('text', "")
            if not input_text:
                continue
                
            fut = ex.submit(
                process_single_embedding,
                runtime=runtime,
                key=key,
                text=input_text,
                disable_rewrite=disable_rewrite,
                disable_duration_est=disable_duration_est
            )
            futures[fut] = key

        success_count = 0
        for i, fut in enumerate(as_completed(futures)):
            key = futures[fut]
            try:
                res = fut.result()
                results_map[key] = res["text_embedding"]
                success_count += 1
                
                if i % 10 == 0:
                    print(f">>> Progress: {i}/{total_tasks} | Success: {success_count}")
                    
            except Exception as e:
                print(f">>> Failed processing key {key}: {str(e)}")

    print(f">>> Finished. Total success: {success_count}/{total_tasks}")
    return results_map


def main():
    """
    python dump_text.py --model_path ckpts/tencent/HY-Motion-1.0 \
        --dataset_path motionfix_test.pth.tar \
        --output_path motionfix_test_embeddings.pt \
        --device_ids 0,1 \
    """
    import argparse

    parser = argparse.ArgumentParser(description="HY-Motion Text Embedding Extractor")
    parser.add_argument("--model_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--dataset_path", type=str, required=True, help="")
    parser.add_argument("--output_path", type=str, default="output_embeddings.pt", help="Path to save result dict")
    parser.add_argument("--device_ids", type=str, default=None, help="GPU device ID list")
    
    # LLM 相关参数
    parser.add_argument("--prompt_engineering_model_path", type=str, default=None)
    parser.add_argument("--prompt_engineering_host", type=str, default=None)
    parser.add_argument("--disable_rewrite", action="store_true", help="Disable text rewriting")
    
    args = parser.parse_args()

    # 1. 初始化 Runtime
    cfg = os.path.join(args.model_path, "config.yml")
    ckpt = os.path.join(args.model_path, "latest.ckpt")
    
    device_ids = None
    if args.device_ids:
        device_ids = [int(x.strip()) for x in args.device_ids.split(",")]

    print(">>> Initializing T2MRuntime...")
    runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        device_ids=device_ids,
        disable_prompt_engineering=args.disable_rewrite, # 如果只提取embedding，可能不需要预估时长
        prompt_engineering_host=args.prompt_engineering_host,
        prompt_engineering_model_path=args.prompt_engineering_model_path,
    )

    breakpoint()

    dataset_dict_raw = joblib.load(args.dataset_path)

    embeddings_map = run_parallel_embedding_tasks(
        runtime=runtime,
        dataset_dict=dataset_dict_raw,
        disable_rewrite=args.disable_rewrite,
        disable_duration_est=True,
    )

    print(f">>> Saving {len(embeddings_map)} embeddings to {args.output_path}")
    torch.save(embeddings_map, args.output_path)

if __name__ == "__main__":
    main()