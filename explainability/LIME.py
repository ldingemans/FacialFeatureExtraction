import pandas as pd
import numpy as np
import pickle
import pymc3 as pm
import matplotlib.pyplot as plt

import os
import sys
PATH_TO_GESTALTMATCHER_DIR = ""
sys.path.append(PATH_TO_GESTALTMATCHER_DIR)
sys.path.append(os.path.join(PATH_TO_GESTALTMATCHER_DIR, 'GestaltEngine-FaceCropper-retinaface'))
from get_feature_vector import gm_preprocess_image, convert_representation_to_feature_vector
from predict_ensemble import predict_memory


def predict_lime_image(X_test, classifier_args):
    """
    Get predictions for a given image, or filepath. Can be used by LIME to get predictions for perturbated images
    
    Parameters
    ----------
    X_test:
        Can be a numpy array of normalized VGG-Face feature vectors, a list of file paths or a single file path
        
    Returns
    -------
    predictions: numpy array
        Predictions per class
    """
    
    failed_images = []
    
    if type(X_test) == str:
        X_test = gm_preprocess_image(X_test)
    elif type(X_test) == list:
        X_test_temp = []
        for i, file in enumerate(X_test):
            try:
                X_test_temp.append(gm_preprocess_image(file))
            except ValueError as e:
                if 'Face could not be detected.' in str(e):
                    X_test_temp.append(np.zeros((1,224,224,3)))
                    failed_images.append(i)
                else:
                    raise(e)
        X_test = np.array(X_test_temp)

    classifier_dict = classifier_args

    pm_model = classifier_dict['model']
    trace = classifier_dict['trace']
    X_shared = classifier_dict['X_shared']
    gm_models = classifier_dict['gm_models']
    device = classifier_dict['device']

    img_test = []

    for i in range(X_test.shape[0]):
        img_test.append(convert_representation_to_feature_vector(predict_memory(gm_models, device, X_test[i])))
   
    img_test = np.array(img_test)

    X_shared.set_value(np.array(img_test))
    ppc = pm.sample_posterior_predictive(trace, samples=5000, model=pm_model, var_names=['y_model', 'probabilities'], progressbar=False)
    predictions = ppc['probabilities'].mean(axis=0)
    predicted_classes = np.argmax(np.exp(predictions).T / np.sum(np.exp(predictions), axis=1), axis=0)
    if len(failed_images) > 0:
        predicted_classes[np.array(failed_images)] = -1
        predictions[np.array(failed_images),:] = np.nan
        print("There were images in which a face was not detected and therefore the image was not processed. Predictions are np.nan for that instance, please check.")
    return predictions


def get_segmentation_mask(image):
    """
    Generate a random mask for an image

    Parameters
    ----------
    image: numpy array
        Image for which the random mask needs to be generated

    Returns
    -------
    Generated random mask
    """
    img_length = image.shape[0]
    chunk_length = int(img_length / 9)

    total_pixels = int(img_length * img_length)
    n_masks = int(np.ceil(total_pixels / chunk_length))
    shifted_mask = np.zeros(np.random.randint(chunk_length), dtype=int)
    shifted_mask = np.append(shifted_mask, np.repeat(list(range(1, n_masks + 1)), chunk_length))
    shifted_mask = shifted_mask[:total_pixels].reshape(img_length, img_length)
    shifted_mask = np.repeat(shifted_mask, chunk_length, axis=0)
    shifted_mask = shifted_mask[np.random.randint(chunk_length):, :]
    shifted_mask = shifted_mask[:img_length, :img_length]

    return np.array(shifted_mask, dtype=int)


def draw_heatmap(explanation, n_syndromes):
    """
    Draw the heatmap of the LIME explanations for a single instance
    
    Parameters
    ----------
    explanation: LIME explanation
        The generated explanation instance
    n_syndromes: list 
        List of syndromes, to be used to convert indices to syndrome names
    """
    ind =  explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(ind)
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    plt.title(n_syndromes[ind] + ', LIME score:' + str(np.round(explanation.score,2)))
    plt.imshow(heatmap, cmap = 'Blues', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.imshow(temp, alpha=0.5)
    plt.axis('off')
    plt.show()
    return


def draw_top_heatmaps(df_results, n_syndromes):
    """
    Draw the average heatmaps for the top n (default 10) predictions for classes in n_syndromes
    
    Parameters
    ----------
    dF_results: pd.DataFrame
        The generated results file by the main softmax_regression.py
    n_syndromes: list 
        List of syndromes
    """
    y_true = df_results.y_true.explode().to_numpy(dtype=int)
    y_pred = df_results.predicted_classes.explode().to_numpy(dtype=int)

    df_exploded = pd.DataFrame()
    df_exploded['label_true'] = n_syndromes[y_true].reset_index(drop=True)
    df_exploded['y_true'] = y_true
    df_exploded['y_pred'] = y_pred
    df_exploded['lime_explanations'] = df_results.loc[:, 'explanations'].reset_index(drop=True)
        
    predictions = df_results.predictions.reset_index(drop=True)
    predictions = pd.DataFrame.from_dict(dict(zip(predictions.index, predictions.values)))
    predictions = predictions.T
    preds_syn = 'preds_' + pd.Series(n_syndromes)
    predictions.columns = preds_syn
    df_exploded = pd.concat([df_exploded.reset_index(drop=True), predictions],axis=1, ignore_index=True)
    col_names = ['syndrome', 'y_true', 'y_pred', 'lime_explanations']
    col_names.extend(list(range(4, 43)))
    df_exploded.columns = col_names

    list_of_heatmaps = df_exploded.loc[:, 'lime_explanations'].explode().to_list()
    heatmaps = []
    for explanation_ in list_of_heatmaps:
        try:
            ind = explanation_.top_labels[0]
            dict_heatmap = dict(explanation_.local_exp[ind])
            temp_heatmap = np.vectorize(dict_heatmap.get)(explanation_.segments)
            temp_heatmap[temp_heatmap == None] = np.nan
            heatmaps.append(temp_heatmap)
        except:
            heatmaps = np.zeros((112,112, 3))
            continue
    heatmap = np.nanmean(heatmaps, axis=0)
    heatmap = np.array(heatmap, dtype=float)

    max_heatmap = heatmap.max()

    mean_face = plt.imread("background_facial_image.png")
    plt.imshow(heatmap, cmap='seismic_r', vmin=-max_heatmap, vmax=max_heatmap)
    ax = plt.imshow(mean_face, alpha=0.3)
    ax.axes.xaxis.set_visible(False)
    plt.yticks([], [])
    plt.savefig("heatmap.png", dpi=300)
    plt.show()
    return
