import torch

def get_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Grouping parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": 1e-5,
            "weight_decay": 1e-2,
        },
        {
            "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
        {
            "params": model.decoder.parameters(),
            "lr": 1e-4, # Decoder learns faster
            "weight_decay": 1e-2,
        },
    ]
    
    return torch.optim.AdamW(optimizer_grouped_parameters)