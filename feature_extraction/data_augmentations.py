import cv2
import albumentations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil


def get_augmented_image(filename):
    """

    Parameters
    ----------
    filename: str
        path to image to augment

    Returns
    -------
    transformed: numpy array
        augmented image
    """
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = albumentations.Compose([
        albumentations.RandomBrightnessContrast(),
        albumentations.MedianBlur(),
        albumentations.HorizontalFlip(),
        albumentations.ColorJitter(),
        albumentations.Spatter(p=0.1),
        albumentations.ToGray(p=0.1)
    ])

    transformed = transform(image=image)
    return transformed["image"]


def augment_images(labels, input_files, output_directory, target_augmentations=400):
    """

    Parameters
    ----------
    target_augmentations
    labels: list/numpy array/pandas series
        the corresponding labels for the input files
    input_files: list/numpy array/pandas series
        the paths to the input images to be augmented
    output_directory: str
        output directory for augmented images

    Returns
    -------
    output_paths: list
        the paths to augmented images
    """
    df_syndromes = pd.DataFrame()
    df_syndromes['labels'], df_syndromes['path_to_file'] = labels, input_files

    output_paths, filenames_parent, labels_augmented = [], [], []

    for syndrome in np.unique(df_syndromes.labels):
        df_this_syndrome = df_syndromes[df_syndromes.loc[:, "labels"] == syndrome].reset_index(drop=True)
        path_to_files = pd.Series()
        n_repeat_this_df = int(np.ceil(target_augmentations / len(df_this_syndrome.labels)))
        path_to_files = pd.concat([df_this_syndrome.path_to_file] * n_repeat_this_df, ignore_index=True)

        assert len(path_to_files) >= target_augmentations

        Path(os.path.join(output_directory, 'augmented', syndrome)).mkdir(parents=True, exist_ok=True)

        for i in range(target_augmentations):
            aug_image = get_augmented_image(path_to_files[i])
            filename = path_to_files[i].split('\\')[-1]
            extension = filename[-4:]
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_directory, 'augmented', syndrome,
                                       filename.replace(extension, '_' + str(i) + extension))
            cv2.imwrite(output_path, aug_image)
            output_paths.append(output_path)
            filenames_parent.append(filename[:-4])
            labels_augmented.append(syndrome)
    return output_paths, filenames_parent, labels_augmented
