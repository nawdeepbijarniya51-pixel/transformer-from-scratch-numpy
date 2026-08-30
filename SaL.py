import numpy as np
import json
import os

def save_model(model, save_dir, config, epoch=None):
    os.makedirs(save_dir, exist_ok=True)

    # collect every named parameter into one dict
    named_params = dict(model.named_parameters())

    # save weights
    weights_path = os.path.join(save_dir, "model_weights.npz")
    np.savez(weights_path, **named_params)

    # save config (architecture hyperparameters) so the model can be rebuilt later
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_dir}/")
    print(f"  - weights: {weights_path} ({len(named_params)} arrays)")
    print(f"  - config:  {config_path}")


def load_model(model, save_dir):
    weights_path = os.path.join(save_dir, "model_weights.npz")
    loaded = np.load(weights_path)

    named_params = dict(model.named_parameters())
    for name, param in named_params.items():
        param[...] = loaded[name]   # in-place copy, preserves the array object

    print(f"Loaded {len(named_params)} arrays from {weights_path}")