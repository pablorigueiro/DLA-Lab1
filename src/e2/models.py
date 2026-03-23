import torch.nn as nn
from torchvision.models import get_model

def build_model(config):
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    freeze_backbone = config["model"]["freeze_backbone"]

    if model_name == "resnet18":
        weights = "DEFAULT" if pretrained else None
        model = get_model('resnet18',weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Keep de backbone (feature extractor) fixed. Do not compute gradients and update parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Re-enable classifier if backbone is frozen: unfreeze our final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(config["system"]["device"])
    return model