import os
import torch

from .models import build_model
from .training import build_loss, build_optimizer, train_one_epoch, evaluate
from .utils import set_seed, count_parameters


def run_experiment(config, dl_train, dl_test):
    set_seed(config["system"]["seed"])
    device = config["system"]["device"]

    model = build_model(config)
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)

    run_name = config["logging"]["run_name"]
    epochs = config["training"]["epochs"]

    print(f"\nRunning experiment: {run_name}")
    print(f"Model parameters: {count_parameters(model):,}")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_val_loss = None
    best_val_report = None

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{run_name}_best.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, epoch=epoch + 1, device=device
        )

        val_loss, val_acc, val_report = evaluate(
            model, dl_test, criterion, device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_val_report = val_report

            torch.save(model.state_dict(), checkpoint_path)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    print("\nTraining finished")
    print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint saved to: {checkpoint_path}")

    return {
        "run_name": run_name,
        "model_name": config["model"]["name"],
        "pretrained": config["model"]["pretrained"],
        "freeze_backbone": config["model"]["freeze_backbone"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr": config["training"]["lr"],
        "optimizer": config["training"]["optimizer"],
        "weight_decay": config["training"]["weight_decay"],
        "num_parameters": count_parameters(model),
        "checkpoint_path": checkpoint_path,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_val_report": best_val_report,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
    }