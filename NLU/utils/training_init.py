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
    
    def get_optimizer(self, optimizer_name, model, lr, eps = 1e-8):
        optimizer = None
        if optimizer_name == "AdamW":
            optimizer = AdamW(model.parameters(),
                              lr = lr,
                              eps = eps)
        if optimizer != None:
            return optimizer
        else:
            print("None optimizer name")
            
    def get_metric(self, task_name):
        glue = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]
        super_glue = ["axb", "axg", "boolq", "cb", "copa", "multirc", "record", "wic", "wsc"]
        if task_name in glue:
            return load('glue', task_name)
        if task_name in super_glue:
            return load("super_glue", task_name)
        
        raise Exception("No such dataset. Check your dataset name")