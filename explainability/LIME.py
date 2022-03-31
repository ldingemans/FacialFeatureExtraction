import pandas as pd
import numpy as np
from deepface import DeepFace
from deepface.DeepFace import build_model
import pickle
import pymc3 as pm
from tqdm import tqdm
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import cv2

def predict_image(X_test):
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
        X_test = get_norm_image(X_test)
    elif type(X_test) == list:
        X_test_temp = []
        for i, file in enumerate(X_test):
            try:
                X_test_temp.append(get_norm_image(file))
            except ValueError as e:
                if 'Face could not be detected.' in str(e):
                    X_test_temp.append(np.zeros((1,224,224,3)))
                    failed_images.append(i)
                else:
                    raise(e)
        X_test = np.array(X_test_temp)
        
    img_test = []
    model_vgg = build_model("VGG-Face")
    
    # print('Getting VGG-Face face embeddings..')
    
    for i in range(X_test.shape[0]):
        img_test.append(model_vgg.predict(X_test[i].reshape(1, 224, 224, 3))[0].tolist())
   
    img_test = np.array(img_test)
        
    # print('Getting predictions from Bayesian classifier..')
    try:
        X_shared.set_value(np.array(img_test))
    except:
        with open('model.pickle', 'rb') as buff:
            data = pickle.load(buff)
            model = data['model']
            trace = data['trace']
            X_shared = data['X_shared']
            X_shared.set_value(np.array(img_test))
    ppc = pm.sample_posterior_predictive(trace, samples=5000, model=model, var_names=['y_model', 'probabilities'], progressbar=False)
    predictions = ppc['probabilities'].mean(axis=0)
    predicted_classes = np.argmax(np.exp(predictions).T / np.sum(np.exp(predictions), axis=1), axis=0)
    if len(failed_images) > 0:
        predicted_classes[np.array(failed_images)] = -1
        predictions[np.array(failed_images),:] = np.nan
        print("There were images in which a face was not detected and therefore the image was not processed. Predictions are np.nan for that instance, please check.")
    return predictions

def get_segmentation_mask(image):
    """
    Obtain the segmented image of a facial picture
    
    Parameters
    ----------
    image: numpy array
        Image to process
        
    Returns
    -------
    final_mask: numpy array
        The mask for the 11 segments   
    """
    import mediapipe
    import skimage
    
    drawingModule = mediapipe.solutions.drawing_utils
    faceModule = mediapipe.solutions.face_mesh
     
    image = skimage.img_as_ubyte(image)
    with faceModule.FaceMesh(static_image_mode=True) as face:
        results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []
    raw_landmarks = []
    for f in results.multi_face_landmarks[0].landmark:
        landmarks.append([f.x, f.y, f.z])
        raw_landmarks.append([f.x * image.shape[1], f.y * image.shape[0], f.z])
    landmarks, raw_landmarks = pd.DataFrame(landmarks),pd.DataFrame(raw_landmarks) 
    
    forehead_marks = [162, 70, 63, 105, 66, 107, 9, 336, 296, 334, 293,300, 389]
    right_eye_marks= [226, 113, 225, 224, 223, 222, 221, 189, 244,233, 232, 231,230, 229, 228, 31]
    left_eye_marks= [464, 413, 441, 442, 443, 444, 445, 342, 446,448, 449, 450, 451, 452, 453]
    nose_marks = [9, 55, 189, 244, 128, 114, 217, 126, 129, 98, 97, 2, 326, 327, 294, 358, 279, 429, 437, 343, 357, 464, 413, 285]
    filtrum_marks = [129, 203, 206, 216, 186,39, 37, 0, 267, 269, 410,436, 426, 423, 358, 294, 327, 326, 2, 97, 98, 64]
    mouth_marks = [186,39, 37, 0, 267, 269, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57]
    chin_marks = [57,58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]
    right_eyebrow_marks = [162, 70, 63, 105, 66, 107, 9, 55, 189, 221, 222, 223, 224, 225, 113, 226, 162]
    left_eyebrow_marks = [9, 336, 296, 334, 293,300, 389, 342, 445, 444, 443, 442, 441, 413, 285]
    left_cheek_marks = [464, 453, 452, 451, 450, 449, 448, 446, 342, 389,  288, 287, 410, 426,423, 358, 279, 429, 437, 343, 357]
    right_cheek_marks = [233, 232, 231,230, 229, 228, 31, 226, 162, 127, 234, 93, 132, 58, 57, 186, 216, 206, 203, 129, 126, 217, 114, 128, 244]

    all_marks = []
    all_marks.append(['forehead', forehead_marks])    
    all_marks.append(['right_eye', right_eye_marks])    
    all_marks.append(['left_eye', left_eye_marks])    
    all_marks.append(['nose', nose_marks])    
    all_marks.append(['filtrum', filtrum_marks])    
    all_marks.append(['mouth', mouth_marks])    
    all_marks.append(['chin', chin_marks])    
    all_marks.append(['right_eyebrow', right_eyebrow_marks])    
    all_marks.append(['left_eyebrow', left_eyebrow_marks])     
    all_marks.append(['right_cheek', right_cheek_marks])    
    all_marks.append(['left_cheek', left_cheek_marks])       
    
    all_coors = []
    
    for i in range(len(all_marks)):
        if i == 0:
            all_coors.append(['forehead', raw_landmarks.iloc[162, 0], 0])
        
        for coor in all_marks[i][1]:
            all_coors.append([all_marks[i][0], raw_landmarks.iloc[coor, 0], raw_landmarks.iloc[coor, 1]])
        if i == 0:
            all_coors.append(['forehead',raw_landmarks.iloc[389, 0], 0])
    
    all_coors = pd.DataFrame(all_coors)

    polygon_masks = []
    
    final_mask = np.zeros((224,224))
    
    for part in np.unique(all_coors.iloc[:,0]):
        polygon_masks.append(skimage.draw.polygon2mask((224, 224),  all_coors.loc[all_coors.iloc[:,0] == part, [1,2]].to_numpy()).T)
    
    for i in range(1,len(polygon_masks)+1):
        final_mask[polygon_masks[i-1]] = i
    
    return np.array(final_mask,dtype='int64')

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
    from sklearn.metrics import confusion_matrix
    import scipy
    
    y_true = df_results.y_true_vgg.explode().to_numpy(dtype=int)
    y_pred = df_results.vgg_classes.explode().to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    
    df_exploded = pd.DataFrame()
    df_exploded['label_true'] = df_results.label_true_vgg.explode().reset_index(drop=True)
    df_exploded['y_true'] = y_true
    df_exploded['y_pred'] = y_pred
    df_exploded['lime_explanations'] = df_results.lime_explanations.explode().reset_index(drop=True)
        
    predictions = df_results.vgg_pred.explode().reset_index(drop=True)
    predictions = pd.DataFrame.from_dict(dict(zip(predictions.index, predictions.values)))
    predictions = predictions.T
    preds_syn = 'preds_' + pd.Series(n_syndromes)
    predictions.columns = preds_syn
    df_exploded = pd.concat([df_exploded, predictions],axis=1)
    df_exploded = df_exploded[df_exploded.lime_explanations.astype(str).str.len()> 10]
    
    N_TOP = 10
    fig, axs = plt.subplots(3,6, figsize=(33,15))
        
    axs = axs.flatten()
    
    for y,pred_syn in enumerate(preds_syn):
        df_top = df_exploded[df_exploded.label_true == n_syndromes[y]].reset_index(drop=True)
        df_top = df_top.iloc[df_top.nlargest(N_TOP, pred_syn).index,:].reset_index(drop=True)
        
        assert len(df_top) == N_TOP
        
        all_coefs = pd.DataFrame()
        for i in range(len(df_top)):
            all_coefs = pd.concat([all_coefs, pd.DataFrame(df_top.lime_explanations[i].local_exp[1])],axis=0).reset_index(drop=True)
        
        dict_heatmap_avg = all_coefs.groupby(0).mean().to_dict()[1]
        
        ppv = cm[y,y] / cm[:,y].sum()
        sens = cm[y,y] / cm[y,:].sum()
        
        heatmap = np.vectorize(dict_heatmap_avg.get)(df_exploded.lime_explanations[0].segments) 
        axs[y].set_title(n_syndromes[y] + ', PPV:' + str(np.round(ppv, 2)) + ', Sensitivity:' + str(np.round(sens, 2)))
        sm = axs[y].imshow(scipy.ndimage.uniform_filter(heatmap, size=1), cmap = 'Blues', vmin  = -heatmap.max(), vmax = heatmap.max())
        fig.colorbar(sm, ax=axs[y])
        axs[y].axis('off')
    plt.show()
    return
