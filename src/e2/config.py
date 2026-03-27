config = {
    "model": {
        "name": "resnet18",
        "num_classes": 43,
        "pretrained": True,
        "freeze_backbone": False
    },
    "training": {
        "epochs": 10,
        "batch_size": 1024,
        "lr": 1e-3,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "weight_decay": 0.0,
        "momentum": 0.9
    },
    "data": {
        "num_workers": 2
    },
    "system": {
        "seed": 42,
        "device": "cuda"
    },
    "logging": {
        "run_name": "baseline"
    }
}