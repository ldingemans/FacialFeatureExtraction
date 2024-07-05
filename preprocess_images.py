from feature_extraction.process_images import process_image_list
from feature_extraction.data_augmentations import augment_images
from glob import glob
import pandas as pd


if __name__ == '__main__':
    # code below assumes that IMAGE_PATH is a directory in which subdirectories with the name of the labels are
    # these subdirectories contain the images. i.e. IMAGE_PATH\syndrome_1\image.jpg
    IMAGE_PATH = r''

    all_images = glob(IMAGE_PATH + r'\*\*.png')
    all_images.extend(glob(IMAGE_PATH + r'\*\*.jpg'))

    labels = pd.Series(all_images).str.split('\\').str[-2]
    filenames = pd.Series(all_images).str.split('\\').str[-1].str[:-4]

    df_data = pd.DataFrame()
    df_data['filename'] = filenames
    df_data['filename_unaugmented_parent'] = 'is_parent'
    df_data['gene'] = labels
    df_data['mp_vector'], df_data['facenet_vector'], df_data['vgg_vector'], df_data['qmagface_vector'] = '', '', '', ''
    X_vgg, X_mp, X_facenet, X_qmagface = process_image_list(all_images)
    for i in range(len(df_data)):
        df_data.at[i, 'mp_vector'] = X_mp[i, :]
        df_data.at[i, 'facenet_vector'] = X_facenet[i, :]
        df_data.at[i, 'vgg_vector'] = X_vgg[i, :]
        df_data.at[i, 'qmagface_vector'] = X_qmagface[i, :]

    all_images_augmented, all_images_augmented_parents, augmented_labels = \
        augment_images(labels, all_images, IMAGE_PATH, 400)

    df_augmented = pd.DataFrame()
    df_augmented['filename'] = pd.Series(all_images_augmented).str.split('\\').str[-1].str[:-4]
    df_augmented['filename_unaugmented_parent'] = all_images_augmented_parents
    df_augmented['gene'] = augmented_labels
    df_augmented['mp_vector'], df_augmented['facenet_vector'], \
    df_augmented['vgg_vector'], df_augmented['qmagface_vector'] = '', '', '', ''
    X_vgg_aug, X_mp_aug, X_facenet_aug, X_qmagface_aug = process_image_list(all_images_augmented)
    for i in range(len(df_augmented)):
        df_augmented.at[i, 'mp_vector'] = X_mp_aug[i, :]
        df_augmented.at[i, 'facenet_vector'] = X_facenet_aug[i, :]
        df_augmented.at[i, 'vgg_vector'] = X_vgg_aug[i, :]
        df_augmented.at[i, 'qmagface_vector'] = X_qmagface_aug[i, :]

    df_data = pd.concat([df_data, df_augmented], axis=0).reset_index(drop=True)
    df_data.to_pickle('df_preprocessed_data.pickle')

