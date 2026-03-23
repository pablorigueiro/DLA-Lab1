from .models import build_model
from .training import build_loss, build_optimizer, train_one_epoch, evaluate
from .utils import set_seed, count_parameters

def run_experiment(config, dl_train, dl_test):
    set_seed(config["system"]["seed"])
    device = config["system"]["device"]

    model = build_model(config)
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)

    print(f"\nRunning experiment: {config['logging']['run_name']}")
    print(f"Model parameters: {count_parameters(model):,}")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    val_report = None

    for epoch in range(config["training"]["epochs"]):
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

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    print("\nTraining finished")
    print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_report": val_report,
    }