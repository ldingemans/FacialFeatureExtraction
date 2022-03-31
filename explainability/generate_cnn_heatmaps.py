from deepface.basemodels.VGGFace import loadModel
from keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface.commons import functions, realtime, distance as dst
from deepface import *
from glob import glob
from tqdm import tqdm
    
def get_norm_image(img_path):
    """
    Preprocess the image for VGG-Face, using MTCNN (detect face, alignment, etc)
    
    Parameters
    ----------
    img_path: str
        Path to the image to process
        
    Returns
    -------
    img_tensor: numpy array
        The preprocessed image in array form
    """
    classifier = loadModel()
    input_shape_x, input_shape_y = functions.find_input_shape(classifier)
    
    img = functions.preprocess_face(img = img_path
       		, target_size=(input_shape_y, input_shape_x)
       		, enforce_detection = True
       		, detector_backend = 'mtcnn'
       		, align = True)
    
    img_tensor = functions.normalize_input(img = img, normalization = 'base')
    return img_tensor

def generate_activation_maps(path_to_image):
    """
    Generate activation maps for a speficic image for VGG-Face
    
    Parameters
    ----------
    path_to_image: str
        Path to the image to process
    """
    classifier = loadModel()
    
    img_tensor = get_norm_image(path_to_image)
    
    layer_outputs = [layer.output for layer in classifier.layers[1:]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
    
    layer_names = []
    for layer in classifier.layers[1:]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        
    count = 0
    
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        count += 1
        if ('conv' not in layer_name):
            continue
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        if count == 1:
            images_per_row = 3
        else:
            images_per_row = int(np.round(np.sqrt(n_features)))
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                if count < 32:
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1 / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.axis('off')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig("Activation_map_" + str(count) + '_' + layer_name + '.png', dpi=300, bbox_inches='tight')    
    return
  
