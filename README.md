<h1>Comparison of facial feature extraction methods</h1>

The code of the paper in which we compare facial feature extraction methods in the diagnosis of individuals with genetic disorders.

With the code in this repository, it is possible to recreate all results of our study (given the input data) and furthermore, use the different facial feature extraction methods for your own images/dataset.

<h2>Prerequisites</h2>

1) First install the needed dependencies using pip:

```
pip install pandas numpy pymc3 theano sklearn arviz tqdm deepface imblearn matplotlib seaborn scikit-image lime albumentations
```

2) Download/clone this repository.

<h2>Overall structure of repository</h2>

The main analysis can be run from the `softmax_regression.py` script in the root directory. If you are interested in the different feature extraction methods, please see the 
`process_images.py` script in the `feature_extraction` directory. Do note that processing of images with the hybrid model is not possible in Python and needs its own virtual machine: if you are interested, please contact me and I will supply the needed files.

For `GestaltMatcher-arc` the code and models are not freely available, please see https://www.gestaltmatcher.org/ for the procedure to get access. Then update `PATH_TO_GESTALTMATCHER_DIR` to point it to where the model is.

For `QMagFace`, please see https://github.com/pterhoer/QMagFace and clone that repository in the root directory.

In the `explainability` folder, you can find the needed scripts to generate all figures and tables displayed in our paper. Furthermore the code to generate activation maps for `GestaltMatcher-arc` (or another CNN) is there, as is the code to generate the LIME heatmaps.

<h2>Retraining the model on your own images</h2>

To repeat the analysis on your own dataset and generate the corresponding tables/figures in our paper, please first run the ``preprocess_images.py`` script to obtain the augmented images. Then, just run the scripts in `softmax_regression.py` and you are all set!
