import torch
import numpy as np
import os
from transformers import AdamW
import random
from evaluate import load
"""
multi GPU then
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
"""

class InitProcessor:
    def __init__(self, seed = 42):
        seed_val = seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
    def get_device(self, config, use_gpu = True, CUDA_VISIBLE_DEVICES = "0, 1, 2"):
        if use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            device = torch.device("cuda:0")
            config.device = "cuda:0"
            return config
   
        device = torch.device("cpu")
        config.device = "cpu"
        return config
    
    def get_optimizer(self, optimizer_name, model, lr, eps = 1e-8, alpha = 1):
        optimizer_1 = None
        optimizer_2 = None
        param_1 = [
            {
                "params": [p for n, p in model.named_parameters() if "query" in n or "hidden" in n]
            }
        ]
        param_2 = [
            {
                "params": [p for n, p in model.named_parameters() if "attention" in n]
            }
        ]
        if optimizer_name == "AdamW":
            optimizer_1 = AdamW(param_1,
                                lr = lr * alpha,
                                eps = eps)
            optimizer_2 = AdamW(param_2,
                                lr = lr,
                                eps = eps)
        if optimizer_1 != None and optimizer_2 != None:
            return optimizer_1, optimizer_2
        else:
            print("None optimizer name")
            
