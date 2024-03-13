import torch
def validation(config, model, batch, TASK_NAME, labels = None):
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
            
    with torch.no_grad():
        outputs = model(input_ids = b_input_ids, 
                        attention_mask = b_input_attention,         
                        token_type_ids = b_input_type,
                        labels = b_input_label)
        del b_input_ids
        del b_input_attention
        del b_input_type
        b_input_label.to("cpu")
        return outputs, b_input_label