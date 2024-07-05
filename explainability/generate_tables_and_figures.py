import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'..')
import seaborn as sns
import matplotlib.pyplot as plt


def demographics_table(excel_file_path, latex_file_path):
    def calculate_sex_distribution_no_words(sex_series):
        male_count = (sex_series == 'm').sum()
        female_count = (sex_series == 'f').sum()
        total_count = male_count + female_count
        male_percentage = (male_count / total_count) * 100 if total_count else 0
        female_percentage = (female_count / total_count) * 100 if total_count else 0
        return f"{male_count} ({male_percentage:.0f}%) / {female_count} ({female_percentage:.0f}%)"
    # Load the Excel file
    df = pd.read_excel(excel_file_path)

    # Define a dictionary for renaming the labels
    rename_dict = {
        'CORNELIA DE LANGE SYNDROME 1': 'Cornelia de Lange Syndrome',
        'DEAF1_AD': 'DEAF1 (AD)',
        'DEAF1_AR': 'DEAF1 (AR)',
        'KdVs': 'KANSL1',
        'NICOLAIDES BARAITSER SYNDROME': 'Nicolaides Baraitser Syndrome',
        'SATB1_PTV': 'SATB1 (PTV)',
        'SATB1_missense': 'SATB1 (missense)',
        'SMITH-LEMLI-OPITZ SYNDROME': 'SLO Syndrome',
        'WILLIAMS-BEUREN SYDNROME': 'Williams-Beuren Syndrome'
    }

    # Rename columns as per requirements
    df = df.rename(columns={'Website': 'Genetic syndrome', 'gender': 'Sex', 'age_photo_rounded': 'Age'})
    df['Genetic syndrome'] = df['Genetic syndrome'].replace(rename_dict)

    # Group by 'Genetic syndrome' and calculate the necessary statistics
    grouped = df.groupby('Genetic syndrome').agg(
        Number_of_individuals=('Genetic syndrome', 'size'),
        Sex_distribution=('Sex', calculate_sex_distribution_no_words),
        Age_median=('Age', lambda x: round(x.median(), 1))
    ).reset_index()

    # Convert the updated DataFrame to a LaTeX table
    latex_table = grouped.to_latex(
        index=False,
        header=['Genetic syndrome', 'Number of individuals', 'Sex distribution (%)', 'Age (median in years)'],
        column_format='lccc'
    )

    # Replace this with the desired path for your LaTeX table file
    with open(latex_file_path, 'w') as f:
        f.write(latex_table)


def print_results(df_results):
    """
    Generate the results table with the results
    
    Parameters
    ----------
    df_results : pandas dataframe 
        Dataframe with the results from the softmax regression (see softmax_regression.py)
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss
    from scipy.stats import sem, t
    from sklearn.preprocessing import label_binarize

    # Function to compute 95% Confidence Interval
    def confidence_interval(data, confidence=0.95):
        n = len(data)
        m = np.mean(data)
        std_err = sem(data)
        interval = std_err * t.ppf((1 + confidence) / 2., n - 1)
        return m, m - interval, m + interval

    # Create an empty DataFrame to store the final results
    result_table = pd.DataFrame(
        index=[f'Accuracy (n = {size} per class)' for size in [5, 10, 25]] + [f'Log Loss (n = {size} per class)' for
                                                                               size in [5, 10, 25]],
        columns=['Hybrid model', 'MediaPipe', 'FaceNet', 'VGGFace2', 'QMagFace', 'GestaltMatcher-arc'])

    all_labels = list(range(39))

    # Loop through different augmentation sizes
    for size in [5, 10, 25]:

        training_size = size * (len(np.unique(df_results['classes'].explode())))
        # Filter DataFrame based on 'classes' to get only rows that belong to each augmentation size
        filtered_df = df_results[df_results['train_filenames'].str.len() == training_size]

        # Group by 'model'
        grouped = filtered_df.groupby('model')

        # Loop through each group to calculate metrics and CIs
        for name, group in grouped:

            acc_list = []
            logloss_list = []

            # Loop through each fold
            for idx, row in group.iterrows():
                # Calculate Accuracy for the fold
                acc = accuracy_score(row['y_true'], row['classes'])
                acc_list.append(acc)

                # Calculate Log Loss for the fold
                binarized_y_true = label_binarize(row['y_true'], classes=all_labels)

                # Calculate Log Loss for the fold
                logloss = log_loss(binarized_y_true, row['pred'])
                logloss_list.append(logloss)

            # Calculate 95% CI for Accuracy
            acc, acc_lower, acc_upper = confidence_interval(acc_list)

            # Calculate 95% CI for Log Loss
            logloss, logloss_lower, logloss_upper = confidence_interval(logloss_list)

            specific_name = {
                'hybrid': 'Hybrid model',
                'mp': 'MediaPipe',
                'facenet': 'FaceNet',
                'vgg': 'VGGFace2',
                'qmagface': 'QMagFace',
                'gm': 'GestaltMatcher-arc'
            }.get(name, name)

            result_table.at[
                f'Accuracy (n = {size} per class)', specific_name] = f"{acc:.2f} [{acc_lower:.2f}-{acc_upper:.2f}]"
            result_table.at[
                f'Log Loss (n = {size} per class)', specific_name] = f"{logloss:.2f} [{logloss_lower:.2f}-{logloss_upper:.2f}]"

    result_table.to_excel("results_softmax.xlsx")
    result_table.to_latex("results_softmax.tex",bold_rows=True)
    print(result_table.to_latex(bold_rows=True))
    return


def get_confusion_matrices(df_results, correct_labels):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Filter to only include 'gm' model
    gm_df = df_results[df_results['model'] == 'gm']

    # Initialize figure
    fig = plt.figure(figsize=(35, 20))

    # Manually add axes at [left, bottom, width, height]
    ax1 = fig.add_axes([0.1, 0.62, 0.35, 0.35])  # Top-left
    ax2 = fig.add_axes([0.55, 0.62, 0.35, 0.35])  # Top-right
    ax3 = fig.add_axes([0.325, 0.15, 0.35, 0.35])  # Center of the second row

    axes = [ax1, ax2, ax3]

    # Loop through different augmentation sizes
    for idx, size in enumerate([5, 10, 25]):

        training_size = size * (len(np.unique(gm_df['y_true'].explode())))
        # Filter DataFrame based on 'classes' to get only rows that belong to each augmentation size
        filtered_gm_df = gm_df[gm_df['train_filenames'].str.len() == training_size]

        # Initialize confusion matrix to all zeros
        all_labels = sorted(
            list(set(np.concatenate(gm_df['y_true'].values))))  # Assumes all labels are present in the 'y_true' column
        num_labels = len(all_labels)
        cm = np.zeros((num_labels, num_labels), dtype=np.int)

        # Loop through each row in the filtered DataFrame to update the confusion matrix
        for _, row in filtered_gm_df.iterrows():
            true_labels = np.array(row['y_true'])
            pred_labels = np.array(row['classes'])
            row_cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
            cm += row_cm

        # Plot the confusion matrix
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=correct_labels, yticklabels=correct_labels, ax=ax)
        ax.set_title(f'Confusion Matrix for GestaltMatcher-arc model (n = {size} per class)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Show the plot
    plt.savefig(r"fig_confusion_matrices.pdf", dpi=300)
    plt.show()
