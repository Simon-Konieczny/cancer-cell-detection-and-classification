import timm
import torch.nn as nn

def get_model(model_name="efficientnetv2_rw_m", num_classes=3, input_size=640):
    if model_name.startswith("swin_tiny") or model_name.startswith("maxvit_tiny"):
        model = timm.create_model(model_name,
                                pretrained=True,
                                num_classes=num_classes,
                                in_chans=3,
                                img_size=input_size,
                                drop_rate=0.3,
                                drop_path_rate=0.2)
    else:
        model = timm.create_model(model_name,
                                pretrained=True,
                                num_classes=num_classes,
                                in_chans=3,
                                drop_rate=0.3,
                                drop_path_rate=0.2)
    return model