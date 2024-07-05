import pandas as pd
import numpy as np

from bayesian_models.bayes_log_reg import bayes_logistic_reg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler

from explainability.generate_cnn_heatmaps import generate_activation_maps
from explainability.LIME import draw_heatmap, draw_top_heatmaps, get_segmentation_mask, predict_lime_image
from glob import glob
import os
from random import sample
from explainability.generate_tables_and_figures import print_results, get_confusion_matrices
from lime import lime_image
import pickle
import traceback
import sys

PATH_TO_GESTALTMATCHER_DIR = ""
sys.path.append(os.path.join(PATH_TO_GESTALTMATCHER_DIR))
sys.path.append(os.path.join(PATH_TO_GESTALTMATCHER_DIR, 'GestaltEngine-FaceCropper-retinaface'))
from get_feature_vector import gm_preprocess_image, preload_files


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


def get_lime_images(df_data, df_results, syndrome_to_investigate, factorizations, N_SAMPLES = 10, model='gm'):
    skf = StratifiedKFold(n_splits=3)

    models, device, gallery_set_representations, representation_df, model_detect, device_detect = preload_files(cpu=True)

    df_results = df_results[df_results['model'] == model]
    df_results = df_results[df_results['train_filenames'].str.len() == df_results['train_filenames'].str.len().max()].reset_index(drop=True)
    assert len(df_results) == skf.n_splits

    all_files = df_results.test_filenames.explode().reset_index(drop=True)
    all_predictions = df_results.pred.explode().reset_index(drop=True)

    df_files_and_predictions_exploded = pd.concat([all_files, all_predictions], axis=1)
    df_files_and_predictions_exploded_syndrome = df_files_and_predictions_exploded[df_files_and_predictions_exploded.iloc[:,0].str.contains(syndrome_to_investigate.lower())].reset_index(drop=True)

    for i in range(len(df_files_and_predictions_exploded_syndrome)):
        #get prediction for that specific syndrome
        df_files_and_predictions_exploded_syndrome.loc[i, 'pred'] = df_files_and_predictions_exploded_syndrome.loc[i, 'pred'][np.argmax(factorizations == syndrome_to_investigate)]

    top_correctly_predicted =  df_files_and_predictions_exploded_syndrome.loc[df_files_and_predictions_exploded_syndrome['pred'].astype(float).nlargest(N_SAMPLES).index, 'test_filenames']

    df_all_explan = pd.DataFrame()

    for train_index, test_index in skf.split(df_unaugmented.gene, df_unaugmented.gene):
        df_train = df_augmented[df_augmented['filename_unaugmented_parent'].isin(
            df_unaugmented.loc[train_index, 'filename'])].groupby('gene').sample(N_SAMPLES).reset_index(
            drop=True)
        y_train = df_train['gene']

        for n in range(len(y_train)):
            y_train[n] = np.argmax(factorizations == y_train[n])

        X_train = []
        for i in range(len(df_train)):
            X_train.append(df_train.loc[i, model + '_vector'])
        X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=int)

        X_test, y_test, y_test_filename = [], [], []
        for i in range(len(df_unaugmented)):
            if i in test_index and np.mean(pd.isna(df_unaugmented.loc[i, model + '_vector']) == False) == 1:
                X_test.append(df_unaugmented.loc[i, model + '_vector'])
                y_test.append(df_unaugmented.loc[i, 'gene'])
                y_test_filename.append(df_unaugmented.loc[i, 'filename'])
        X_test, y_test, y_test_filename = np.array(X_test, dtype=np.float32), np.array(y_test), np.array(
            y_test_filename)
        for n in range(len(y_test)):
            y_test[n] = np.argmax(factorizations == y_test[n])

        X_train, X_test, y_train, y_test, ind_remove_test = \
            remove_empties(X_train, X_test, y_train, y_test)

        predictions, predicted_classes, trace, summ_trace, X_shared, pm_model = bayes_logistic_reg(X_train,
                                                                                                   y_train,
                                                                                                   X_test,
                                                                                                   advi=False,
                                                                                                   N_CORES=1,
                                                                                                   prior_inclusion_prob=0.1,
                                                                                                   target_accept=0.99,
                                                                                                   tune_steps=1000)

        chosen_indices_top_predictions = np.nonzero(pd.Series(y_test_filename).isin(top_correctly_predicted))
        y_test_this_fold_of_investigated_syndrome = y_test_filename[chosen_indices_top_predictions]

        # Assuming df_data_paths is a pandas DataFrame with 'identifier' and 'path_to_file' columns
        # and y_test_filename is a list of filenames to search for.
        y_test_filename_full = []

        df_data_paths = pd.read_excel(path_to_data_file)
        df_data_paths['identifier'] = df_data_paths['identifier'].str[:-4].str.lower()
        for full_filename in y_test_this_fold_of_investigated_syndrome:
            filename = full_filename[:-4]
            if 'deaf1_' in filename:
                filename = filename[9:]
            # Try to find the full path using the entire filename
            match = df_data_paths[df_data_paths['identifier'] == filename]['path_to_file']
            if match.empty:
                # If no match found, split the filename and try again
                parts = filename.split('_')
                match = df_data_paths[df_data_paths['identifier'] == '_'.join(parts[1:])]['path_to_file']
            if not match.empty:
                # If a match is found, add it to the full filename list
                y_test_filename_full.append(match.iloc[0])
            else:
                # If no match is found after all attempts, raise an error
                raise ValueError(f"No match found for {filename} after all attempts.")

        classifier_dict = {
            'model' : pm_model,
            'trace' : trace,
            'X_shared' : X_shared,
            'gm_models': models,
            'device': device
        }

        explanations = []
        print(y_test_filename_full)
        for file_path in y_test_filename_full:
            exp_face = []
            local_pred_face = []
            for m in range(100):
                segmentation_fn = get_segmentation_mask
                explainer = lime_image.LimeImageExplainer(verbose=False, feature_selection='lasso_path')
                try:
                    explanation = explainer.explain_instance(gm_preprocess_image(file_path, model_detect, device_detect, cpu=True), predict_lime_image,
                                                             top_labels=len(factorizations), num_samples=100,
                                                             batch_size=100, segmentation_fn=segmentation_fn, classifier_args=classifier_dict)
                    exp_face.append(explanation)
                    local_pred_face.append(explanation.local_pred[0])
                except:
                    print(traceback.format_exc())
            explanations.append([file_path, exp_face, local_pred_face])
        df_explan = pd.DataFrame(explanations)
        df_explan.columns = ['file_path', 'explanations', 'lime_pred']
        df_explan['predicted_classes'] = predicted_classes[chosen_indices_top_predictions].astype(int)
        df_explan['y_true'] = y_test[chosen_indices_top_predictions]
        df_explan['y_true'] = df_explan['y_true'].astype(int)
        df_explan['predictions'] = ''
        assert len(df_explan) == len(predictions[chosen_indices_top_predictions])
        for z in range(len(df_explan)):
            df_explan.at[z, 'predictions'] = predictions[chosen_indices_top_predictions][z]
        df_all_explan = pd.concat([df_all_explan, df_explan], axis=0)
        df_all_explan.to_pickle('gestaltmatcher_lime.pickle')


if __name__ == '__main__':
    models = ['gm', 'hybrid', 'mp', 'facenet', 'vgg', 'qmagface']

    try:
        df_data = pd.read_pickle('df_preprocessed_data.pickle')
        for i in range(len(df_data)):
            if 'SATB1' in df_data.loc[i,'gene']:
                if 'missense' in df_data.loc[i,'filename']:
                    df_data.loc[i, 'gene'] = 'SATB1_missense'
                elif 'ptv' in df_data.loc[i,'filename']:
                    df_data.loc[i, 'gene'] = 'SATB1_ptv'
                else:
                    ValueError(i)
            if 'DEAF' in df_data.loc[i, 'gene']:
                if 'ar' in df_data.loc[i, 'filename']:
                    df_data.loc[i, 'gene'] = 'DEAF1_AR'
                elif 'ad' in df_data.loc[i, 'filename']:
                    df_data.loc[i, 'gene'] = 'DEAF1_AD'
                else:
                    ValueError(i)
        assert len(df_data['gene'].unique()) == 39
    except:
        raise ValueError("Please preprocess and augment the images first!")

    df_unaugmented = df_data.loc[df_data['filename_unaugmented_parent'] == 'is_parent', :].reset_index(drop=True)
    df_augmented = df_data.loc[df_data['filename_unaugmented_parent'] != 'is_parent', :]
    df_augmented = df_augmented.dropna().reset_index(drop=True)

    for m in models:
        vec_col = m + '_vector'
        df_augmented = df_augmented[[sum(i) != 0 for i in df_augmented[vec_col]]]
        df_augmented = df_augmented[[~np.isnan(i[0]) for i in df_augmented[vec_col]]]

    count = 0
    if os.path.isfile('softmax_results_.pickle'):
        df_results = pd.read_pickle('softmax_results_.pickle')
        print('result dataframe found and loaded with size ' + str(len(df_results)))
    else:
        df_results = pd.DataFrame()
        df_results['train_filenames'], df_results['test_filenames'] = '', ''
        df_results['model'], df_results['pred'], df_results['classes'], df_results['trace'], df_results['y_true'] = '', '', '', '', ''

    factorizations = pd.Series(pd.factorize(df_unaugmented['gene'])[1])

    for N_SAMPLES in [5, 10, 25]:
        skf = StratifiedKFold(n_splits=3)
        split = 0
        for train_index, test_index in skf.split(df_unaugmented.gene, df_unaugmented.gene):
            split += 1
            for model in models:
                if len(df_results) > count:
                    print("skipping count " + str(count))
                    count += 1
                    continue
                if count % len(models) == 0:
                    df_train = df_augmented[df_augmented['filename_unaugmented_parent'].isin(
                        df_unaugmented.loc[train_index, 'filename'])].groupby('gene').sample(N_SAMPLES).reset_index(drop=True)
                else:
                    prev_filenames = df_results.loc[len(df_results)-1, 'train_filenames']
                    df_train = df_augmented[df_augmented['filename'].isin(prev_filenames)].reset_index(drop=True)

                y_train = df_train['gene']

                for n in range(len(y_train)):
                    y_train[n] = np.argmax(factorizations == y_train[n])

                X_train = []
                for i in range(len(df_train)):
                    X_train.append(df_train.loc[i, model + '_vector'])
                X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=int)

                X_test, y_test, y_test_filename = [], [], []
                for i in range(len(df_unaugmented)):
                    if i in test_index and np.mean(pd.isna(df_unaugmented.loc[i, model + '_vector']) == False) == 1:
                        X_test.append(df_unaugmented.loc[i, model + '_vector'])
                        y_test.append(df_unaugmented.loc[i, 'gene'])
                        y_test_filename.append(df_unaugmented.loc[i, 'filename'])
                X_test, y_test, y_test_filename = np.array(X_test, dtype=np.float32), np.array(y_test), np.array(y_test_filename)
                for n in range(len(y_test)):
                    y_test[n] = np.argmax(factorizations == y_test[n])

                df_results.at[count, 'train_filenames'] = ''
                df_results.at[count, 'train_filenames'] = df_train['filename'].to_numpy()
                df_results.at[count, 'test_filenames'] = y_test_filename

                X_train, X_test, y_train, y_test, ind_remove_test =\
                    remove_empties(X_train, X_test, y_train, y_test)

                if model == 'hybrid':
                    X_train = np.append(normalize(X_train[:,:340]),normalize(X_train[:,340:]),axis=1)
                    X_test = np.append(normalize(X_test[:,:340]),normalize(X_test[:,340:]),axis=1)
                    scale = StandardScaler()
                    X_train = scale.fit_transform(X_train)
                    X_test = scale.transform(X_test)

                if model == 'mp':
                    scale = StandardScaler()
                    X_train = scale.fit_transform(normalize(X_train))
                    X_test = scale.transform(normalize(X_test))

                predictions, predicted_classes, trace, summ_trace, X_shared, pm_model = bayes_logistic_reg(X_train, y_train, X_test, advi=False, N_CORES=1, prior_inclusion_prob=0.1, target_accept=0.99, tune_steps=1000)

                df_results.at[count, 'pred'], df_results.at[count, 'classes'], df_results.at[count, 'trace'] = predictions, predicted_classes, summ_trace
                df_results.at[count, 'model'] = model
                df_results.at[count, 'y_true'] = np.array(y_test, dtype=int)
                df_results.to_pickle('softmax_results_.pickle')
                count += 1

    get_lime_images(df_data, model='gm')

    #generate tables and figures
    print_results(df_results)
    get_confusion_matrices(df_results, factorizations)

    IMAGE_PATH = os.path.join(os.getcwd(), 'images')
    random_img = sample(glob(IMAGE_PATH +'\\*'),1)[0]
    #generate activation maps for this randomly selected image
    generate_activation_maps(random_img)
    
    #draw heatmap
    draw_top_heatmaps(df_results, factorizations)
