import segmentation_models_pytorch as smp

# ResNet34 U-Net++ for Segmentation
def get_model_resnet():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # Transfer learning
        in_channels=3,                # USG images as RGB
        classes=1,                    # Binary segmentation
        activation=None         # For binary output
    )
    return model

def get_model_resnet_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # Transfer learning
        in_channels=3,                # USG images as RGB
        classes=1,                    # Binary segmentation
        activation=None         # For binary output
    )
    return model

def get_model_resnet_deeplabv3():
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # Transfer learning
        in_channels=3,                # USG images as RGB
        classes=1,                    # Binary segmentation
        activation=None         # For binary output
    )
    return model

# Mit-b3
def get_mit_b3_model():
    model = smp.Unet(
        encoder_name="mit_b3", # Mix Vision Transformer (SegFormer)
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

def get_mit_b3_deeplabv3_model():
    model = smp.DeepLabV3(
        encoder_name="mit_b3", # Mix Vision Transformer (SegFormer)
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

def efficient_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3", # Reliable, high-accuracy encoder
        encoder_weights="imagenet",     # This definitely exists and works
        in_channels=3,                  # USG as RGB
        classes=1,                      # Binary segmentation
        activation=None                 # No activation for BCEWithLogits
    )
    return model