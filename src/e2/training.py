import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def build_loss(config):
    loss_name = config["training"]["loss"]

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    
def build_optimizer(config, model):
    optimizer_name = config["training"]["optimizer"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config["training"].get("momentum", 0.9)
        return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def train_one_epoch(model, dl, criterion, optimizer, epoch='Unknown', device=device):
    model.train()
    losses = []
    predictions = []
    gts = []

    for xs, ys in tqdm(dl, desc=f'Training epoch {epoch}', leave=True):
        xs = xs.to(device)
        ys = ys.to(device)

        optimizer.zero_grad()
        logits = model(xs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        predictions.append(preds.detach().cpu().numpy())
        gts.append(ys.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    acc = accuracy_score(np.hstack(gts), np.hstack(predictions))

    return avg_loss, acc

def evaluate(model, dl, criterion, device=device):
    model.eval()
    losses = []
    predictions = []
    gts = []
    with torch.no_grad():
        for xs, ys in tqdm(dl, desc='Evaluating', leave=False):
            xs = xs.to(device)
            ys = ys.to(device)

            logits = model(xs)
            loss = criterion(logits, ys)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.detach().cpu().numpy())
            gts.append(ys.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    acc = accuracy_score(np.hstack(gts), np.hstack(predictions))
    report = classification_report(
        np.hstack(gts),
        np.hstack(predictions),
        zero_division=0,
        digits=3
    )

    return avg_loss, acc, report
