import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model
from tqdm.notebook import tqdm

def build_feature_extractor(model_name, pretrained=True, device="cuda"):
    weights = "DEFAULT" if pretrained else None
    model = get_model(model_name, weights=weights)
    if hasattr(model, "fc"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        else:
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown classifier structure for {model_name}")
    model = model.to(device)
    model.eval()
    return model, feature_dim

def extract_features(dataloader, model, device):
    feats = []
    classes = []
    model.eval()
    for ims, cls in tqdm(dataloader):
        ims = ims.to(device)
        with torch.no_grad():
            f = model(ims)
            f = F.normalize(f, p=2, dim=1)
        feats.append(f.cpu())
        classes.append(cls)
    feats = torch.cat(feats, dim=0)
    classes = torch.cat(classes, dim=0)
    return feats, classes