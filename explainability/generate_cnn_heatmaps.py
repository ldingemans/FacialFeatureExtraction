import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface.commons import functions, realtime, distance as dst
from deepface import *
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn  # Import nn module
from PIL import Image


def get_feature_maps(model, input_tensor):
    """
    Function to get feature maps for each layer of the given model.

    Args:
    model (torch.nn.Module): The neural network model.
    input_tensor (torch.Tensor): The input tensor for the model.

    Returns:
    dict: A dictionary where keys are layer names and values are corresponding feature maps.
    """

    feature_maps = {}

    model.eval()

    def hook_fn(module, input, output):
        # Use the module's name as the key
        for name, mod in model.named_modules():
            if mod == module and 'Conv_' in  name:
                feature_maps[name] = output.detach()
                break

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    model(input_tensor)

    for hook in hooks:
        hook.remove()
    return feature_maps


def generate_activation_maps(models, path_to_preprocessed_image, save_dir):
    """
    Generate activation maps for a speficic image for VGG-Face
    
    Parameters
    ----------
    models: list
        List of the three GestaltMatcher-arc bayesian_models
    path_to_preprocessed_image: str
        Path to the preprocessed image to process
    save_dir: str
        Path to output directory
    """
    import os
    import torch
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image
    import math

    input_tensor = Image.open(path_to_preprocessed_image).convert('RGB')
    convert_tensor = transforms.ToTensor()
    input_tensor = convert_tensor(input_tensor).unsqueeze(0)


    for z, model in enumerate(models):
        # 3. Generate feature maps
        feature_maps = get_feature_maps(model, input_tensor)

        # 4. Visualize the feature maps
        def visualize_feature_maps(feature_maps):
            for layer_name, fmap in feature_maps.items():
                batch, channels, height, width = fmap.shape
                total_feature_maps = fmap.size(1)

                n_cols = int(math.ceil(math.sqrt(total_feature_maps)))
                n_rows = int(math.ceil(total_feature_maps / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
                fig.suptitle(f'Feature maps for layer: {layer_name}', fontsize=16)

                axes_flat = axes.flatten()

                for i in range(n_rows * n_cols):
                    if i < total_feature_maps:
                        row = i // n_cols
                        col = i % n_cols
                        ax = axes[row, col] if n_rows > 1 else axes[col]
                        ax.imshow(fmap[0, i, :, :].cpu())
                
                for ax in axes_flat:
                    ax.axis('off')
                filename = str(z) + '_' + model._get_name() + '_' + layer_name + '.png'
                plt.suptitle(f'Feature maps for layer: {layer_name}')
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.clf()
                plt.close()
                # plt.show()

        # Call the visualization function
        visualize_feature_maps(feature_maps)
    return
  
