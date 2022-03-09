from deepface.basemodels.VGGFace import loadModel
from keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface.commons import functions, realtime, distance as dst
from deepface import *
from glob import glob
from tqdm import tqdm

def calculate_accuracies_lfw(path_to_lfw_database=r"C:\Users\manz184215\scikit_learn_data\lfw_home\lfw"):
    """
    Calculate the classification accuracy in the Labeled Faces in the Wild (LFW) database.
    
    
    Parameters
    ----------
    path_to_lfw_database: str
        Path to the database of LFW. Can be downloaded using sklearn's fetch_lfw_people(min_faces_per_person=30, funneled=False) function

    Returns
    -------
    acc_197: float
        Accuracy on the LFW dataset using the selected 197 features
    acc_2622: float
        Accuracy on the LFW dataset using the full 2622 feature vector
    """
    from sklearn import svm
    from sklearn.datasets import fetch_lfw_pairs
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    
    lfw_people = fetch_lfw_people(min_faces_per_person=30, funneled=False)
    
    list_of_files = []
    for target in lfw_people.target_names:
        for file in glob(path_to_lfw_database + "\\" + target.replace(' ', '_') + '\\*'):
            list_of_files.append([str(target.replace(' ', '_')), file])
    
    list_of_files = pd.DataFrame(list_of_files)
    
    X = []
    y = []
        
    for q in tqdm(range(len(list_of_files))):
        X.append(DeepFace.represent(list_of_files.iloc[q,1],enforce_detection=False, detector_backend='mtcnn'))
        y.append(list_of_files.iloc[q,0])
        
    X_lfw = np.array(X)
    y_lfw = pd.factorize(np.array(y))[0]
    
    y_pred = []
    y_real = []
                   
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X_lfw, y_lfw)
    total_tqdm = skf.get_n_splits(X_lfw, y_lfw)
    for train_index, test_index in tqdm(skf.split(X_lfw, y_lfw), total=total_tqdm):
        X_train, X_test = X_lfw[train_index], X_lfw[test_index]
        y_train, y_test = y_lfw[train_index], y_lfw[test_index]
        
        param_grid = {'C': [1e-5, 1e-3, 1, 1e3, 1e5]}
        
        clf = GridSearchCV(
            svm.SVC(), param_grid, cv=5, scoring='accuracy'
            )
        clf.fit(X_train, y_train)
        y_pred.extend(clf.predict(X_test))
        y_real.extend(y_test)
    
    y_real_2622, y_pred_2622 = y_real[:], y_pred[:]
    
    acc_2622 = np.mean(np.array(y_real_2622) == np.array(y_pred_2622))
    
    VGG_facial_parts = np.load("facial_part_features.npy")
    X_lfw = X_lfw[:, VGG_facial_parts]
    
    y_pred = []
    y_real = []
                   
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X_lfw, y_lfw)
    total_tqdm = skf.get_n_splits(X_lfw, y_lfw)
    for train_index, test_index in tqdm(skf.split(X_lfw, y_lfw), total=total_tqdm):
        X_train, X_test = X_lfw[train_index], X_lfw[test_index]
        y_train, y_test = y_lfw[train_index], y_lfw[test_index]
        
        param_grid = {'C': [1e-5, 1e-3, 1, 1e3, 1e5]}
        
        clf = GridSearchCV(
            svm.SVC(), param_grid, cv=5, scoring='accuracy'
            )
        clf.fit(X_train, y_train)
        y_pred.extend(clf.predict(X_test))
        y_real.extend(y_test)
    
    acc_197 = np.mean(np.array(y_real) == np.array(y_pred))
    return acc_197, acc_2622

def process_facial_parts(seperate_vectors):
    """
    Generate tSNE plots for facial parts, run logistic regression for prediction, get essential regression coefficients

    Returns
    -------
    coefs_filtered: numpy array/pandas dataframe
        Features with non-zero regression coefficients for facial part logistic regression
    """
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    
    seperate_vectors = pd.DataFrame(seperate_vectors)
    
    #below is without full_face
    seperate_vectors = seperate_vectors[seperate_vectors.iloc[:, 1] != 'full_face']
    
    df_variance = pd.DataFrame(columns=seperate_vectors.iloc[:,1].unique())
    
    for part in list(df_variance.columns):
        df_variance[part] = seperate_vectors[seperate_vectors.iloc[:,1] == part].var()
        plt.hist(df_variance[part], label=part)
        
    plt.legend()
    plt.show()
    
    X = seperate_vectors.iloc[:,2:].to_numpy()
    y = pd.factorize(seperate_vectors.iloc[:,1].to_numpy())[0]
        
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)

    plt.figure()
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=seperate_vectors.iloc[:,1].to_numpy(),
        legend="full",
    )
    
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    total_tqdm = skf.get_n_splits(X, y)
    
    y_pred = []
    y_real_test = []
                            
    for train_index, test_index in tqdm(skf.split(X, y), total=total_tqdm):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_real_test.extend(y_test)
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1e2, random_state=1, max_iter=1000000)
        clf.fit(X_train, y_train)
        y_pred.extend(clf.predict(X_test))

    print(np.sum(np.array(y_pred) != np.array(y_real_test)))
    print(np.mean(np.array(y_pred) == np.array(y_real_test)))

    clf = LogisticRegression(penalty='l1', solver='liblinear', C=1e2, random_state=1, max_iter=1000000)
    clf.fit(X,y)
    coefs = pd.DataFrame(clf.coef_)
    coefs.index = pd.factorize(seperate_vectors.iloc[:,1].to_numpy())[1]
    
    coefs_filtered = coefs.T
    coefs_filtered = coefs_filtered.loc[:,["right_eye", "left_eye", "mouth", "nose", "forehead"]]
    coefs_filtered = coefs_filtered.loc[coefs_filtered.sum(axis=1) != 0,:]
    
    coefs_filtered = abs(coefs_filtered).reset_index(drop=True)
    np.save("facial_part_features.npy", list(coefs_filtered.index))
    return coefs_filtered 
    
def crop_facial_image(image, landmark_1, landmark_2, raw_landmarks):
    """
    Crop a specific image with two landmarks, to for instance crop out the mouth and pad the rest of the image with zeros
    
    Parameters
    ----------
    image: numpy array
        The facial image
    landmark_1:
        First landmark to crop from
    landmark_2:
        Second landmark to crop from
        
    Returns
    -------
    black_img: numpy array
        The image of specific facial part with padded zeros
    [y_min,y_max,x_min,x_max]: list
        A list of the used coordinates for cropping
    """
    x_coors = raw_landmarks.iloc[[landmark_1,landmark_2],0]
    y_coors = raw_landmarks.iloc[[landmark_1,landmark_2],1]
    
    x_min, x_max, y_min, y_max = int(x_coors.min()),int(x_coors.max()),int(y_coors.min()),int(y_coors.max())
    
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
        
    if (landmark_1 == 70) and (landmark_2 == 332):
        y_min, y_max,x_min,x_max = 0,y_max,0,224
        
    black_img = np.zeros((224,224,3))
    
    black_img[y_min:y_max,x_min:x_max] = image[y_min:y_max,x_min:x_max]
    plt.imshow(black_img)
    plt.show()    
    return black_img, [y_min,y_max,x_min,x_max]
    
def extract_facial_parts(path_to_arrays):
    """
    Extract the facial parts from the saved preprocessed images to disk (so after running save_norm_image for each image)
    
    Parameters
    ----------
    path_to_arrays: str
        Path to saved files from save_norm_image function
    
    Returns
    ----------
    seperate_vectors: numpy array
        Array with the VGG-Face feature vector for each specific facial part
    """
    import cv2
    import mediapipe
    import matplotlib.pyplot as plt 
    import imutils
    import scipy
    from skimage import img_as_ubyte
    from glob import glob
    classifier = loadModel()
    
    list_of_files = glob(path_to_arrays + "\\*.npy")
    seperate_vectors = np.zeros((len(list_of_files) * 7, 2624),dtype=object)
    
    indexer = 0
    
    for i, file in tqdm(enumerate(list_of_files),total=len(list_of_files)):
        indexer = i * 7

        image = np.load(file)[0]
        
        drawingModule = mediapipe.solutions.drawing_utils
        faceModule = mediapipe.solutions.face_mesh
         
        with faceModule.FaceMesh(static_image_mode=True) as face:
            results = face.process(cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_BGR2RGB))
        landmarks = []
        raw_landmarks = []
        if results.multi_face_landmarks != None:
            for f in results.multi_face_landmarks[0].landmark:
                landmarks.append([f.x, f.y, f.z])
                raw_landmarks.append([f.x * image.shape[1], f.y * image.shape[0], f.z])
            landmarks, raw_landmarks = pd.DataFrame(landmarks),pd.DataFrame(raw_landmarks) 
            
            plt.imshow(image)
            plt.scatter(raw_landmarks.iloc[:,0], raw_landmarks.iloc[:,1])
            plt.show()
            
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'full_face'
            seperate_vectors[indexer, 2:] = classifier.predict(image.reshape(1, 224,224, 3))
                
            right_eye, coor_right_eye = crop_facial_image(image, 46, 114, raw_landmarks)
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'right_eye'
            seperate_vectors[indexer, 2:] = classifier.predict(right_eye.reshape(1, 224,224, 3))
            
            left_eye, coor_left_eye  = crop_facial_image(image, 276, 452, raw_landmarks)
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'left_eye'
            seperate_vectors[indexer, 2:] = classifier.predict(left_eye.reshape(1, 224,224, 3))
            
            mouth, coor_mouth = crop_facial_image(image, 410, 204, raw_landmarks)
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'mouth'
            seperate_vectors[indexer, 2:] = classifier.predict(mouth.reshape(1, 224,224, 3))
            
            nose, coor_nose = crop_facial_image(image, 165, 349, raw_landmarks) 
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'nose'
            seperate_vectors[indexer, 2:] = classifier.predict(nose.reshape(1, 224,224, 3))
            
            forehead, coor_forehead = crop_facial_image(image, 70, 332, raw_landmarks) 
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'forehead'
            seperate_vectors[indexer, 2:] = classifier.predict(forehead.reshape(1, 224,224, 3))
            
            rest_img = image[:]
            rest_img[coor_right_eye[0]:coor_right_eye[1],coor_right_eye[2]:coor_right_eye[3]] = 0
            rest_img[coor_left_eye[0]:coor_left_eye[1],coor_left_eye[2]:coor_left_eye[3]] = 0
            rest_img[coor_mouth[0]:coor_mouth[1],coor_mouth[2]:coor_mouth[3]] = 0
            rest_img[coor_nose[0]:coor_nose[1],coor_nose[2]:coor_nose[3]] = 0
            rest_img[coor_forehead[0]:coor_forehead[1],coor_forehead[2]:coor_forehead[3]] = 0
            indexer += 1
            seperate_vectors[indexer,0] = file.split('\\')[-1][:-8]
            seperate_vectors[indexer,1] = 'rest_image'
            seperate_vectors[indexer, 2:] = classifier.predict(rest_img.reshape(1, 224,224, 3))
            plt.imshow(rest_img)
            plt.show()
    seperate_vectors = seperate_vectors[seperate_vectors[:,2] != 0]
    pd.DataFrame(seperate_vectors).reset_index(drop=True).to_pickle("vectors_facial_parts.pickle")
    return seperate_vectors

def save_norm_image(img_path, save_file):
    """
    Preprocess an image for VGG-Face and then save it to disk, so we can extract the facial parts later
    
    Parameters
    ----------
    img_path: str
        Path to the image to process
    """
    classifier = loadModel()
    img_tensor = get_norm_image(img_path)
    
    np.save(save_file,img_tensor)
    
    full_processed_vector = DeepFace.represent(img_path, detector_backend='mtcnn')
    test_full_face = np.load(save_file + '.npy')
    
    assert (classifier.predict(test_full_face) == full_processed_vector).all()
    return
    
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
  
