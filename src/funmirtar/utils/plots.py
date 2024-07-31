import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, PrecisionRecallDisplay, precision_recall_curve, auc


def plot_multiple_cdfs_with_medians(
    
    dataframe_with_predictions = [],
    columns_for_top_preds = [],
    columns_for_all_lfc = [],
    title_sufix='',
    num_of_top_preds=16
):
    """
    Plots the Cumulative Density Function (CDF) of log2 fold change (LFC) values with median lines for multiple sets of data.
    Each line represents a different dataset, showing the proportion of LFC values below a given threshold.
    The difference is that values_list_for_all_lfc FC values are plotted as a whole, whereas for dataframes_for_top_preds we pick the top 16 predictions and plot only their LFC.
    It is possible to plot only one group (top 16 or all LFC values) and leave the other empty.

    Parameters:
    @param dataframe_with_predictions: DataFrame with predictions. 
    Required columns: 'miRNA', 'fold_change', and columns_for_top_preds and columns_for_all_lfc.
    Column 'miRNA' contains miRNA names, 'fold_change' contains LFC values, and columns_for_top_preds and columns_for_all_lfc contain predictions.
    @param columns_for_top_preds: List of column names with predictions where top N predictions are plotted
    @param columns_for_all_lfc: List of column names with predictions where all LFC values are plotted
    @param title_sufix: To be used in plot title
    @param num_of_top_preds: Number of top predictions for each miRNA to be plotted
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
        
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("colorblind", len(columns_for_top_preds) + len(columns_for_all_lfc))
    
    # plot all LFCs
    for id, column in enumerate(columns_for_all_lfc):
        color = colors[id]
        plot_lines_with_median(dataframe_with_predictions[column], column, color)

    for id, column in enumerate(columns_for_top_preds):
    
        top_predictions = []
        
        # Get top N predictions for each miRNA
        grouped = dataframe_with_predictions.groupby('miRNA')
        for _, group in grouped:
            top_number_of_preds = group.nsmallest(num_of_top_preds, column)
            top_predictions.append(top_number_of_preds['fold_change'])
        
        # Concatenate the top predictions into a single array
        top_predictions = pd.concat(top_predictions)
        #plot LFCs of top N predictions
        color = colors[id + len(columns_for_all_lfc)]
        plot_lines_with_median(top_predictions, column, color)
    
    plt.xlim(-2, 1)
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
    

def plot_correlation_simple(ax, y_true, y_pred, title_sufix):
    xlow, xhigh, ylow, yhigh = -0.5, 0.5, -0.5, 0.5
    sns.histplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        color='darkred',
    )
    sns.regplot(
        x=y_true, 
        y=y_pred, 
        scatter_kws=dict(alpha=0),
        color='brown',
        ax=ax
    )
    ax.text(
        0.7, 
        0.9, 
        f"pearson, r = {pearsonr(y_true, y_pred)[0]:.2f}",
        ha='center',
        transform=ax.transAxes
    )
    ax.text(
        0.7, 
        0.8, 
        f"r^2 = {r2_score(y_true, y_pred):.2f}",
        ha='center',
        transform=ax.transAxes
    )
    ax.text(
        0.7, 
        0.7, 
        f"spearman = {spearmanr(y_true, y_pred)[0]:.2f}",
        ha='center',
        transform=ax.transAxes
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(False)
    ax.set_xlabel('Log2 Fold Change (LFC)')
    ax.set_ylabel('Predictions')
    ax.text(0.5, 1.05, f'Correlation of predictions and LFC - {title_sufix}',
            horizontalalignment='center', transform=ax.transAxes)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_two_correlations(y_true1, y_pred1, title_sufix1, y_true2, y_pred2, title_sufix2, path=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), dpi=300)
    plot_correlation_simple(ax1, y_true1, y_pred1, title_sufix1)
    plot_correlation_simple(ax2, y_true2, y_pred2, title_sufix2)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{path}_two_correlations.png', dpi=fig.dpi, bbox_inches="tight")
    

def plot_feature_importance(feature_names, feature_importances, num_top_features_to_plot=30, path=''):
    # Create a pandas DataFrame for seaborn
    importance_df = pd.DataFrame({
        'Feature Names': feature_names,
        'Importances': feature_importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importances', ascending=False)
    importance_df = importance_df.head(num_top_features_to_plot)

    # Create the plot
    fig = plt.figure(figsize=(5, 4), dpi=300)
    sns.set(style="whitegrid")  # Nature publications have a clean white background

    palette = sns.cubehelix_palette(start=.5, rot=-.5, n_colors=num_top_features_to_plot, reverse=True, as_cmap=False)
    ax = sns.barplot(x="Importances", y="Feature Names", data=importance_df, palette=palette)

    ax.set_xlabel('Importance (sums to 1)', fontsize=8)
    ax.set_ylabel(f'Features (total {len(feature_importances)})', fontsize=8)
    sns.despine(left=True, bottom=True)
    ax.grid(False)
    ax.tick_params(labelsize=5)

    # Add the exact importance value on each bar
    for i, v in enumerate(importance_df["Importances"]):
        ax.text(v, i, f"{v:.3f}", color='black', va='center', size=5)

    plt.tight_layout()
    plt.show()
    fig.savefig(f'{path}.png', dpi=fig.dpi)
    
    
def plot_prc_with_seeds(
    data, seed_types, methods=[], title='', class_label_column='label', count_thresholds=[1,2,3,4], path=''
):
    markers = {
        'kmer8': 'x',
        'kmer7': 'o',
        'kmer6': 'v',
        'kmer6_bulge': '*',
        'kmer6_bulge_or_mismatch': '^'
    }
    
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(5, 5))
    colors = sns.color_palette("colorblind", len(count_thresholds) + len(methods))
    
    for seed_name in seed_types.keys():
        marker = markers[seed_name]
        for threshold in count_thresholds:
            prec, rec, _, _ = precision_recall_fscore_support(data[class_label_column].values, data[seed_name + "_count_" + str(threshold)].values, average='binary')

            ax.plot(rec, prec, marker, color=colors[threshold - 1],  label=seed_name + "_count_" + str(threshold))

    auc_pr_scores = []
    for idx, meth in enumerate(methods):
        precision, recall, _ = precision_recall_curve(data[class_label_column], data[meth])
        auc_pr = auc(recall, precision)
        auc_pr_scores.append((meth, auc_pr))
        
        PrecisionRecallDisplay.from_predictions(
            data[class_label_column], 
            data[meth], 
            ax=ax, label=f"{meth} \nAUC_PR={auc_pr:.2f}", color=colors[idx]
        )
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Remove top and right lines of the plot box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.grid(False)
    plt.show()
    fig.savefig(f'{path}_prc_with_seeds.png', dpi=fig.dpi, bbox_inches="tight")
    return fig, ax

