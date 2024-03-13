from transformers import get_scheduler
import torch
import torch.nn as nn
import random
import numpy as np
import argparse

from utils import Data, record_training, InitProcessor, validation


parser = argparse.ArgumentParser(description='train')

parser.add_argument('--input_dir', type = str, default = None, required=True)
parser.add_argument('--output_dir', type = str, default = './')
parser.add_argument('--save_dir', type = str, default = './')
#save epoch
#3 means save model each 3 epoch
parser.add_argument('--save', '-s', type = int, default = -1)
parser.add_argument('--epochs', '-e', type = int, default = 100)
parser.add_argument('--batch_size', '-b', type = int, default = 16)
parser.add_argument('--lr_rate', '-lr', type = float, default = 2e-5)
parser.add_argument('--use_gpu', '-g', default = False, action = 'store_true')
parser.add_argument('--task_name', type = str, required = True)
parser.add_argument('--model_type', default = "bert-base-uncased")
parser.add_argument('--method_type', default = "add")
parser.add_argument('--rank', type = int, default = 1)

args = parser.parse_args()
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
SAVE = args.save
SAVE_DIR = args.save_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR_RATE = args.lr_rate
GPU = args.use_gpu
MODEL_TYPE = args.model_type
TASK_NAME = args.task_name
METHOD_TYPE = args.method_type
RANK = args.rank

if "roberta" in MODEL_TYPE:
    from Model import TtuneRobertaForSequenceClassification as Model
    from transformers import RobertaConfig as Config
else:
    print(1)
    from Model import TtuneBertForSequenceClassification as Model
    from transformers import BertConfig as Config
    
def run():
    init = InitProcessor(42)
    ###################################data###################################
    global BATCH_SIZE
    DATA_NAME = TASK_NAME  + "_b:" + str(BATCH_SIZE) + "_lr:" + str(LR_RATE) + "_e:" +  str(EPOCHS) + "_t:" + METHOD_TYPE + "_r:" + str(RANK)
    
    origin_batch = BATCH_SIZE
    if BATCH_SIZE > 32:
        BATCH_SIZE = 32
    
    if TASK_NAME == "mnli" and "roberta" not in MODEL_TYPE:
        train_dataloader, valid_dataloader, valid_mis_dataloader = Data(MODEL_TYPE, TASK_NAME,  INPUT_DIR, BATCH_SIZE)
    else:
        train_dataloader, valid_dataloader = Data(MODEL_TYPE, TASK_NAME, INPUT_DIR,  BATCH_SIZE)
        
        
    metric = init.get_metric(TASK_NAME)
    ###################################model###################################
    config = Config().from_pretrained(MODEL_TYPE)
    
    if TASK_NAME == "stsb":
        config.num_labels = 1
    elif TASK_NAME == "mnli":
        config.num_labels = 3
    else:
        config.num_labels = 2
    config.rank = RANK
    config.type = METHOD_TYPE
    model = Model.from_pretrained(MODEL_TYPE, config = config)

    config = init.get_device(config)
    model.to(config.device)
    
    trainable_param = ["classifier", "vector_"]
    for name, param in model.named_parameters():
        flag = True
        for i in trainable_param:
            if i in name:
                flag = False
        if flag:
            param.requires_grad = False

    
    optimizer = init.get_optimizer(optimizer_name = "AdamW", model = model, lr = LR_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_scheduler(name = 'linear',
                              optimizer = optimizer, 
                              num_warmup_steps = int(total_steps * 0.06),
                              num_training_steps = total_steps)
    
    #학습 데이터를 txt파일로 save하기 위한 tool
    #ex) loss, accuracy, F1 score, training time
    record = record_training(metric = metric,
                             epochs = EPOCHS, 
                             data_len = len(train_dataloader), 
                             OUTPUT_DIR = OUTPUT_DIR, 
                             file_name = DATA_NAME)
    
    
    ################################################
    ################### Training ###################
    ################################################
    accumulation_step = int(origin_batch / BATCH_SIZE)
    for epoch_i in range(0, EPOCHS):
        #record.save_before_paramter(model)
        record.init_epoch(epoch_i)
        model.zero_grad()
        model.train()
        num_batches = 0
        for step, batch in enumerate(train_dataloader):
            try:
                b_input_ids, b_input_attention, b_input_type, b_input_label = batch
            except:
                b_input_ids, b_input_attention, b_input_label = batch
                b_input_type = None
            b_input_ids = b_input_ids.type(torch.LongTensor).to(config.device)
            b_input_attention = b_input_attention.type(torch.LongTensor).to(config.device)
            if b_input_type != None:
                b_input_type = b_input_type.type(torch.LongTensor).to(config.device)
            if TASK_NAME == "stsb":
                b_input_label = b_input_label.type(torch.FloatTensor).to(config.device)
            else:
                b_input_label = b_input_label.type(torch.LongTensor).to(config.device)
                
                
            outputs = model(input_ids = b_input_ids, 
                            attention_mask = b_input_attention, 
                            token_type_ids = b_input_type,
                            labels = b_input_label)
            
            loss = outputs[0].mean() / accumulation_step
            loss.backward()
            num_batches += 1
            
            #accumulation
            if num_batches % accumulation_step == 0 or accumulation_step == 0:
                record.save_loss(float(loss))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            del b_input_ids
            del b_input_attention
            del b_input_type
            del b_input_label
            torch.cuda.empty_cache()
            
        record.record_epoch_loss()
        #record.save_parameter_change(model)
        #record.record_parameter_change(model, epoch_i)    

        if SAVE != -1 and epoch_i % SAVE == 0:
            torch.save(model.state_dict(), SAVE_DIR + '/model_%d_'%(epoch_i) + DATA_NAME)
        
        
        model.eval()
        record.init_validation()
        for batch in valid_dataloader:
            outputs, label_ids = validation(config, model, batch, TASK_NAME)
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            record.get_score(logits, label_ids, config.num_labels)
            torch.cuda.empty_cache()
        record.record_metric()
        
        if TASK_NAME == "mnli" and "roberta" not in MODEL_TYPE:
            record.init_validation()
            for batch in valid_mis_dataloader:
                outputs, label_ids = validation(config, model, batch, TASK_NAME)
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()
            
                record.get_score(logits, label_ids, config.num_labels)
                torch.cuda.empty_cache()
            record.record_metric()
    if SAVE != -1:    
        torch.save(model.state_dict(), 
                   SAVE_DIR + '/model_final_' + DATA_NAME)
    
    
if __name__ == "__main__":
    run()
