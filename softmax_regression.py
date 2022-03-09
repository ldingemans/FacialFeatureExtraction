import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 
from feature_extraction.process_images import process_image_list
from models.bayes_log_reg import bayes_logistic_reg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from explainability.generate_cnn_heatmaps import generate_activation_maps, save_norm_image, extract_facial_parts, process_facial_parts
from explainability.generate_tables_and_figures import confusion_matrices, print_results, bar_chart_supplement
from glob import glob
import os
from random import sample

def remove_empties(X_train, X_test, y_train, y_test):
    """
    Remove empty rows in arrays, due to uncorrectly or undetected faces
    
    Parameters
    ----------
    X_train : numpy array
        The training data
    X_test: numpy array
        The validation/testing data
    y_train: numpy array
        The training labels
    y_train: numpy array
        The testing labels
        
    Returns
    -------
    X_train : numpy array
        The training data
    X_test: numpy array
        The validation/testing data
    y_train: numpy array
        The training labels
    y_train: numpy array
        The testing labels
    ind_remove_test: numpy array
        Indices of removed rows in test data
    """
    y_train = y_train[~np.isnan(X_train[:,0])]
    X_train = X_train[~np.isnan(X_train[:,0])]
    ind_remove_test = np.isnan(X_test[:,0])
    y_test = y_test[~np.isnan(X_test[:,0])]
    X_test = X_test[~np.isnan(X_test[:,0])]
    return X_train, X_test, y_train, y_test, ind_remove_test

if __name__ == '__main__':
    PATH_TO_HYBRID_VECTORS = ''
    N_FOLDS = 10 #number of folds for cross validation
    N_RESAMPLE = 5 #number of times to rerun the random under/oversampling analysis
    
    file_paths = ['']
    
    labels = np.array(['DDX3X'] * 10 + ['KANSL1'] * 10 + ['ANKD11'] * 10)
    
    labels = pd.factorize(labels)[0]
    
    y_full_vgg, y_full_mediapipe, y_full_hybrid = labels,labels,labels
    X_full_vgg, X_full_mediapipe = process_image_list(file_paths)
    
    try:
        X_full_hybrid = pd.read_picle(PATH_TO_HYBRID_VECTORS)
    except:
        X_full_hybrid = np.random.uniform(size=(len(file_paths), 468))
    
    results = pd.DataFrame(columns=['vgg_pred','vgg_classes', 'vgg_trace',
                                    'mp_pred', 'mp_classes', 'mp_trace',
                                    'hybrid_pred', 'hybrid_classes', 'hybrid_trace', 
                                    'y_true_vgg', 'y_true_mp', 'y_true_hybrid',
                                    'ind_remove_test_vgg','ind_remove_test_mp', 'ind_remove_test_hybrid'])
    
    
    count = 0
    y_series = pd.Series(labels).value_counts()
    
    for p in range(2): #both with the full 2622 feature vector and the 197 selected one
        if p == 1:
            VGG_facial_parts = np.load("facial_part_features.npy")
            X_full_vgg = X_full_vgg[:, VGG_facial_parts]
        for N_SAMPLES in [20, 40]: #under/oversample to both 20 and 40  
            for z in range(N_RESAMPLE): #and another loop to run this analysis N_RESAMPLE times
                ind_oversamp = list((y_series[(y_series < N_SAMPLES).sort_index()]).index)
                ind_undersamp = list((y_series[(y_series > N_SAMPLES).sort_index()]).index)
                
                oversampling_dict = {}
                undersampling_dict = {}
                
                for i in range(20):
                    if i in ind_oversamp:
                        oversampling_dict[i] = N_SAMPLES
                    elif i in ind_undersamp:
                        undersampling_dict[i] = N_SAMPLES
                                    
                ind_sampling = np.array(range(len(X_full_vgg))).reshape(-1,1)
                
                sampler = RandomUnderSampler(undersampling_dict)
                ind_sampling,y_sampled = sampler.fit_resample(ind_sampling,y_full_vgg)
                sampler = RandomOverSampler(oversampling_dict)
                ind_sampling,y_sampled = sampler.fit_resample(ind_sampling,y_sampled)
                
                ind_sampling = ind_sampling.flatten()
                
                X_sampled_vgg, y_sampled_vgg = X_full_vgg[ind_sampling], y_full_vgg[ind_sampling]
                X_sampled_mp, y_sampled_mp = X_full_mediapipe[ind_sampling], y_full_mediapipe[ind_sampling]   
                X_sampled_hybrid, y_sampled_hybrid = X_full_hybrid[ind_sampling], y_full_hybrid[ind_sampling]
                
                assert (y_sampled_mp == y_sampled_vgg).all()
                assert (y_sampled_vgg == y_sampled_hybrid).all()
                
                skf = StratifiedKFold(n_splits=N_FOLDS)
                for train_index, test_index in skf.split(X_sampled_vgg, y_sampled_vgg):
                     count += 1
                     
                     X_train_vgg, X_test_vgg = X_sampled_vgg[train_index], X_sampled_vgg[test_index]
                     y_train_vgg, y_test_vgg = y_sampled_vgg[train_index], y_sampled_vgg[test_index]
                     
                     X_train_mp, X_test_mp = X_sampled_mp[train_index], X_sampled_mp[test_index]
                     y_train_mp, y_test_mp = y_sampled_mp[train_index], y_sampled_mp[test_index]
                     
                     X_train_hybrid, X_test_hybrid = X_sampled_hybrid[train_index], X_sampled_hybrid[test_index]
                     y_train_hybrid, y_test_hybrid = y_sampled_hybrid[train_index], y_sampled_hybrid[test_index]
            
                     assert (y_test_mp == y_test_vgg).all()
                     assert (y_test_vgg == y_test_hybrid).all()
                     
                     X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid, ind_remove_test_hybrid = remove_empties(X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid)
                     X_train_mp, X_test_mp, y_train_mp, y_test_mp, ind_remove_test_mp = remove_empties(X_train_mp, X_test_mp, y_train_mp, y_test_mp)
                     X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg, ind_remove_test_vgg = remove_empties(X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg)
                     
                     X_train_hybrid = np.append(normalize(X_train_hybrid[:,:340]),normalize(X_train_hybrid[:,340:]),axis=1)
                     X_test_hybrid = np.append(normalize(X_test_hybrid[:,:340]),normalize(X_test_hybrid[:,340:]),axis=1)
                     scale = StandardScaler()
                     X_train_hybrid = scale.fit_transform(X_train_hybrid)
                     X_test_hybrid = scale.transform(X_test_hybrid)
                         
                     scale = StandardScaler()
                     X_train_mp = scale.fit_transform(normalize(X_train_mp))
                     X_test_mp = scale.transform(normalize(X_test_mp))
            
                     print("VGG shapes" + str(X_train_vgg.shape))
                     print("MP shapes" + str(X_train_mp.shape))
                     print("hybrid shapes" + str(X_train_hybrid.shape))
                     
                     predictions_hybrid, predicted_classes_hybrid, trace, summ_trace_hybrid = bayes_logistic_reg(X_train_hybrid, y_train_hybrid, X_test_hybrid, advi=False, prior_inclusion_prob=0.1, target_accept=0.99, tune_steps=1000)
                     results.at[count, 'hybrid_pred'], results.at[count, 'hybrid_classes'], results.at[count, 'hybrid_trace'] = predictions_hybrid, predicted_classes_hybrid, summ_trace_hybrid
                     
                     predictions_vgg, predicted_classes_vgg, trace, summ_trace_vgg = bayes_logistic_reg(X_train_vgg, y_train_vgg, X_test_vgg, advi=False, prior_inclusion_prob=0.1, target_accept=0.9, tune_steps=1000)
                     results.at[count, 'vgg_pred'], results.at[count, 'vgg_classes'], results.at[count, 'vgg_trace'] = predictions_vgg, predicted_classes_vgg, summ_trace_vgg
            
                     predictions_mp, predicted_classes_mp, trace, summ_trace_mp = bayes_logistic_reg(X_train_mp, y_train_mp, X_test_mp, advi=False, prior_inclusion_prob=0.1, target_accept=0.9, tune_steps=1000)
                     results.at[count, 'mp_pred'], results.at[count, 'mp_classes'], results.at[count, 'mp_trace'] = predictions_mp, predicted_classes_mp, summ_trace_mp
                         
                     results.at[count, 'y_true_mp'] = y_test_mp
                     results.at[count, 'y_true_vgg'] = y_test_vgg
                     results.at[count, 'y_true_hybrid'] = y_test_hybrid
                     
                     results.at[count, 'ind_remove_test_vgg'] = ind_remove_test_vgg
                     results.at[count, 'ind_remove_test_mp'] = ind_remove_test_mp
                     results.at[count, 'ind_remove_test_hybrid'] = ind_remove_test_hybrid
                     
                     results.to_pickle('softmax_results.pickle')
    
    df_results = results.iloc[:int(len(results)/2),:].reset_index(drop=True)
    df_results_197 = results.iloc[int(len(results)/2):,:].reset_index(drop=True)
    
    df_results_20 = df_results.iloc[:int(len(df_results)/2),:].reset_index(drop=True)
    df_results_40 = df_results.iloc[int(len(df_results)/2):,:].reset_index(drop=True)
    
    #generate tables and figures
    bar_chart_supplement(df_results, df_results_197)
    confusion_matrices(df_results_20, df_results_40)
    print_results(df_results_20, df_results_40)
    
    IMAGE_PATH = os.path.join(os.getcwd(), 'images')
    
    random_img = sample(glob(IMAGE_PATH +'\\*'),1)[0]
    #generate activation maps for this randomly selected image
    generate_activation_maps(random_img)
    
    #extract partial facial features (nose/mouth etc) and get the relevant features in the feature fector of VGG-Face    
    for img_path in glob(IMAGE_PATH +'\\*'):
        if 'npy' not in img_path:
            save_norm_image(img_path, img_path)
    seperate_vectors = extract_facial_parts(IMAGE_PATH)
    coefs_filtered = process_facial_parts(seperate_vectors)
