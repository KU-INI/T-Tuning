import torch
def validation(config, model, batch, TASK_NAME, labels = None):
    b_input_ids, b_input_attention = batch
    b_input_ids = b_input_ids.type(torch.LongTensor).to(config.device)
    b_input_attention = b_input_attention.type(torch.LongTensor).to(config.device)
    with torch.no_grad():
        outputs = model.generate(input_ids = b_input_ids,
                                 attention_mask = b_input_attention,
                                 max_new_tokens = 64,
                                 num_beams = 10,
                                 no_repeat_ngram_size = 4,
                                 length_penalty = 0.9,
                                 early_stopping=True)
        del b_input_ids
        del b_input_attention
        return outputs