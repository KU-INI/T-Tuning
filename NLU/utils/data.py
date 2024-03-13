from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

    
def BDataLoader(data, batch = 16, sampler = "random"):
    data = TensorDataset(data['input_ids'], 
                             data['attention_mask'], 
                             data['token_type_ids'], 
                             data['labels'])
    if sampler == "random":
        data_sampler = RandomSampler(data)
    else:
        data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler = data_sampler, batch_size = batch)
    return dataloader
        
def RoBDataLoader(data, batch = 16, sampler = "random"):
    data = TensorDataset(data['input_ids'], 
                         data['attention_mask'], 
                         data['labels'])
    if sampler == "random":
        data_sampler = RandomSampler(data)
    else:
        data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler = data_sampler, batch_size = batch)
    return dataloader
    
def Data(model_type, task_name, path, batch_size):
    train = torch.load(path + task_name + "_train")
    if task_name == "mnli" and "roberta" not in model_type:
        valid_matched = torch.load(path + task_name + "_validation_matched")
        valid_mismatched = torch.load(path + task_name + "_validation_mismatched")
    else:
        valid = torch.load(path + task_name + "_validation")
        
    if "roberta" in model_type:
        train_dataloader = RoBDataLoader(train, batch_size)
        valid_dataloader = RoBDataLoader(valid, batch_size)
        return train_dataloader, valid_dataloader
                
    elif "bert" in model_type:
        train_dataloader = BDataLoader(train, batch_size)
        if task_name == "mnli":
            valid_matched_dataloader = BDataLoader(valid_matched, batch_size)
            valid_mis_dataloader = BDataLoader(valid_mismatched, batch_size)
            return train_dataloader, valid_matched_dataloader, valid_mis_dataloader
        else:
            valid_dataloader = BDataLoader(valid, batch_size)
            return train_dataloader, valid_dataloader
                
                
                