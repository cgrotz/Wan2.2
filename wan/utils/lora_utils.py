# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import torch
import torch
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model

def get_loraconfig(transformer, rank=128, alpha=128, init_lora_weights="gaussian", target_modules=None):
    if target_modules is None:
        target_modules = []
        for name, module in transformer.named_modules():
            # Default heuristic for Wan2.2 models: targeting Linear layers in blocks
            # Excluding face/modulation layers as seen in animate_utils.py 
            # But making it slightly more generic if possible
            if "blocks" in name and isinstance(module, torch.nn.Linear):
               if "face" not in name and "modulation" not in name:
                    target_modules.append(name)
    
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    return transformer_lora_config

def apply_lora(model, lora_path, alpha=1.0, merge_lora=True):
    """
    Applies LoRA adapters to the given model from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to attach adapters to.
        lora_path (str): Path to the LoRA checkpoint.
        alpha (float): Scaling factor for the LoRA adapter (if supported by implementation).
                       Note: PEFT LoraConfig takes alpha, but dynamic scaling might need more work 
                       if we want to adjust it at runtime. For now, we assume standard loading.
        merge_lora (bool): Whether to merge the LoRA weights into the base model and unload the adapters.
                           This saves VRAM during inference. Defaults to True.
    """
    if not os.path.exists(lora_path):
        logging.warning(f"LoRA path {lora_path} does not exist. Skipping LoRA loading.")
        return model

    logging.info(f"Loading LoRA from {lora_path}")
    
    # Heuristic: 
    # 1. Define LoraConfig. For now, we reuse the logic from animate_utils which seems tuned for this architecture.
    #    We might need to infer rank/alpha from the checkpoint if possible, strictly speaking.
    #    However, `peft` usually needs the config first. 
    #    Let's try to load the state dict and see if we can guess rank, or just default to 128 as per existing code.
    
    # Load state dict first to inspect
    try:
        if lora_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location="cpu")
            if "state_dict" in lora_state_dict:
                 lora_state_dict = lora_state_dict["state_dict"]
    except Exception as e:
        logging.error(f"Failed to load LoRA checkpoint: {e}")
        return model

    # Create config. 
    # NOTE: Hardcoded rank/alpha = 128 matching animate_utils. 
    # In a perfect world, we'd read this from an accompanying config file.
    lora_config = get_loraconfig(
        transformer=model,
        rank=128, 
        alpha=128
    )
    
    # Inject adapters
    # Inject adapters
    model.add_adapter(lora_config) # WanModel does not support this natively
