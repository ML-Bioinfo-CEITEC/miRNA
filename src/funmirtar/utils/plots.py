import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, PrecisionRecallDisplay


def plot_multiple_cdfs_with_medians(
    dataframes_for_top_preds=[], labels_for_top_preds=[], 
    values_list_for_all_lfc=[], labels_for_all_flc=[], 
    title_sufix='', num_of_top_preds=16
):
    """
    Plots the Cumulative Density Function (CDF) of log2 fold change (LFC) values with median lines for multiple sets of data.
    Each line represents a different dataset, showing the proportion of LFC values below a given threshold.
    The difference is that values_list_for_all_lfc FC values are plotted as a whole, whereas for dataframes_for_top_preds we pick the top 16 predictions and plot only their LFC.
    It is possible to plot only one group (top 16 or all LFC values) and leave the other empty.

    Parameters:
    - dataframes_for_top_preds: List of DataFrames, each containing 'miRNA', predictions (denoted by label), and fold changes (denoted by 'fold_change'). Only top N predictions for each miRNA are picked and their LFC is plotted. 
    - labels_for_top_preds: List of prediction labels, one label for each method
    - values_list_for_all_lfc: List of arrays of LFC values, all values will be plotted
    - labels_for_all_flc: List of labels_for_all_flc for values_list_for_all_lfc, one label for each LFC array
    - title_sufix: To be used in plot title
    - num_of_top_preds: Number of top predictions for each miRNA
    """
    def plot_lines_with_median(predictions, label, color):
        # Compute the CDF
        sorted_values = np.sort(predictions)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        # Compute the median
        median_value = np.median(sorted_values)
        # Plot the CDF
        plt.plot(sorted_values, cdf, linestyle='-', color=color, label=label)
        # Plot the median line
        plt.plot([median_value, median_value], [0, 0.5], linestyle='--', color='black', label='_nolegend_')
        
    plt.figure(figsize=(8, 6))
    # colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    colors = sns.color_palette("colorblind", len(values_list_for_all_lfc) + len(dataframes_for_top_preds))
    
    # plot all LFCs
    for lfc_values, label, color in zip(values_list_for_all_lfc, labels_for_all_flc, colors[:len(values_list_for_all_lfc)]):
        plot_lines_with_median(lfc_values, label, color)
        
    # plot only top N LFCs per miRNA
    for df, label, color in zip(
        dataframes_for_top_preds, labels_for_top_preds, colors[len(values_list_for_all_lfc):]
    ):
        top_predictions = []
        
        # Get top N predictions for each miRNA
        grouped = df.groupby('miRNA')
        for name, group in grouped:
            top_number_of_preds = group.nsmallest(num_of_top_preds, label)
            top_predictions.append(top_number_of_preds['fold_change'])
        
        # Concatenate the top predictions into a single array
        top_predictions = pd.concat(top_predictions)
        #plot LFCs of top N predictions
        plot_lines_with_median(top_predictions, label, color)
    
    plt.xlim(-2, 2)
    plt.xlabel('Log2 Fold Change (LFC)')
    plt.ylabel('Cumulative Density Function (CDF)')
    plt.title(f'CDF of Top {num_of_top_preds} Predicted Targets per miRNA - {title_sufix}')
    plt.legend(loc='lower right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.grid(False)
    plt.show()


def plot_correlation(y_true, y_pred, title_sufix):
    plt.figure(figsize=(5, 5),dpi=300)
    xlow, xhigh, ylow, yhigh = -0.5,0.5,-0.5,0.5
    hexplot = sns.jointplot(
        x=y_true,
        y=y_pred,
        kind='hist',
        color='darkred',
    )
    sns.regplot(
        x=y_true, 
        y=y_pred, 
        scatter_kws=dict(alpha=0),
        color='brown'
    )
    plt.text(
        0.7, 
        0.9, 
        f"pearson, r = {pearsonr(y_true, y_pred)[0]:.2f}",
        # f"r = {r_regression(y_true, y_pred):.2f}",
        ha='center'
    )
    plt.text(
        0.7, 
        0.8, 
        f"r^2 = {r2_score(y_true, y_pred):.2f}",
        ha='center'
    )
    plt.text(
        0.7, 
        0.7, 
        f"spearman = {spearmanr(y_true, y_pred)[0]:.2f}",
        # f"r = {r_regression(y_true, y_pred):.2f}",
        ha='center'
    )

    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # plt.grid(b=None)
    plt.grid(False)
    plt.xlabel('Log2 Fold Change (LFC)')
    plt.ylabel('Predictions')
    # plt.title(f'Correlation of predictions and LFC - {title_sufix}', y=1.5)
    plt.text(0, 1.5, f'Correlation of predictions and LFC - {title_sufix}',
         horizontalalignment='center')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    
def plot_feature_importance(feature_names, feature_importances):
    # Create a pandas DataFrame for seaborn
    importance_df = pd.DataFrame({
        'Feature Names': feature_names,
        'Importances': feature_importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importances', ascending=False)

    # Create the plot
    plt.figure(figsize=(5, 4),dpi=300)
    sns.set(style="whitegrid")  # Nature publications have a clean white background

    ax = sns.barplot(x="Importances", y="Feature Names", data=importance_df, palette='YlOrBr')

    ax.set_xlabel('Importance',fontsize=8)
    ax.set_ylabel('Features',fontsize=8)
    sns.despine(left=True, bottom=True)
    ax.grid(False)
    ax.tick_params(labelsize=5)

    # add the exact importance value on each bar
    for i, v in enumerate(importance_df["Importances"]):
        ax.text(v, i, f"{v:.2f}", color='black', va='center', size=5)

    plt.tight_layout()
    plt.show()
    

def plot_prc_with_seeds(data, seed_types, methods=[], title='', class_label_column='label', count_thresholds=[1,2,3,4]):
    markers = {
        'kmer8': 'x',
        'kmer7': 'o',
        'kmer6': 'v',
        'kmer6_bulge': '*',
        'kmer6_bulge_or_mismatch': '^'
    }
    
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(5, 5))
    # colors = ['brown', 'blue', 'orange', 'pink', 'gray', 'black']
    colors = sns.color_palette("colorblind", len(count_thresholds))
    for seed_name in seed_types.keys():
        marker = markers[seed_name]
        for threshold in count_thresholds:
            prec, rec, _, _ = precision_recall_fscore_support(data[class_label_column].values, data[seed_name + "_count_" + str(threshold)].values, average='binary')

            ax.plot(rec, prec, marker, color=colors[threshold - 1],  label=seed_name + "_count_" + str(threshold))

    for idx, meth in enumerate(methods):
        PrecisionRecallDisplay.from_predictions(
            data[class_label_column], 
            data[meth], 
            ax=ax, label=str(meth), color=colors[idx]
        )
     
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.gcf().set_dpi(300)
    plt.grid(False)
    plt.show()
    return fig, ax
