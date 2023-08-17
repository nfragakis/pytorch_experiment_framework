import torch

def freeze_layers(model, FREEZE_RATIO=0.8):
    n_layers = len([x for x in model.parameters() if x.requires_grad])

    freeze_layers = int(n_layers * FREEZE_RATIO)
    print(f'freezing {freeze_layers} / {n_layers}')

    for i, p in enumerate(
        filter(lambda p: p.requires_grad, model.parameters())
    ):
        if (i < freeze_layers) & (p.requires_grad == True):
            p.requires_grad = False
    return model