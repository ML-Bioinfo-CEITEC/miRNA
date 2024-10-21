import argparse
import time
import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import shap
from deepexperiment.utils import one_hot_encoding_batch
from deepexperiment.interpret import DeepShap
from funmirtar.models.constants import TARGETSCAN_COLUMN_TO_SEQUENCE

def main():
    parser = argparse.ArgumentParser(description='Scan sequences and compute explainability scores.')
    parser.add_argument('--sequence_source_path', type=str, default='../../data/processed/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.pkl', help='Path to the sequence source pickle file.')
    parser.add_argument('--model_path', type=str, default='models/miRBind.h5', help='Path to the model file.')
    parser.add_argument('--explainability_background_data_path', type=str, default='evaluation_set_1_1_CLASH2013_paper.tsv', help='Path to the background data file for explainability.')
    parser.add_argument('--save_explainability_scores_path', type=str, default='../../data/scanned/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.explainability_scores.json', help='Path to save the explainability scores JSON.')
    parser.add_argument('--save_scanning_errors_path', type=str, default='../../data/scanned/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.explainability_scores.scanning_errors.txt', help='Path to save the scanning errors.')
    parser.add_argument('--prediction_threshold', type=float, default=0.0, help='Prediction threshold.')
    parser.add_argument('--id_column', type=str, default='RefSeq ID', help='ID column name in sequence source.')
    parser.add_argument('--mirna_name', type=str, help='Name of miRNA to use from TARGETSCAN_COLUMN_TO_SEQUENCE.')
    args = parser.parse_args()

    PREDICTION_THRESHOLD = args.prediction_threshold
    ID_COLUMN = args.id_column
    SEQUENCE_SOURCE_PATH = args.sequence_source_path
    MODEL_PATH = args.model_path
    EXPLAINABILITY_BACKGROUND_DATA_PATH = args.explainability_background_data_path
    SAVE_EXPLAINABILITY_SCORES_PATH = args.save_explainability_scores_path
    SAVE_SCANNING_ERRORS_PATH = args.save_scanning_errors_path
    MIRNA_NAME = args.mirna_name

    my_miRNA = TARGETSCAN_COLUMN_TO_SEQUENCE[MIRNA_NAME]
    mirna_sequences = [my_miRNA]
    print(f"Scanning for {mirna_sequences}, {MIRNA_NAME}")

    random.seed(42)

    # Load and preprocess the transcript data
    sequence_source_df = pd.read_pickle(SEQUENCE_SOURCE_PATH)
    gene_symbol_to_seq = sequence_source_df[[ID_COLUMN, "sequence"]].set_index(ID_COLUMN).to_dict()['sequence']

    # Load the model and the data
    model = keras.models.load_model(MODEL_PATH)   # Old model from miRBind trained on Ago1 data

    samples = pd.read_csv(EXPLAINABILITY_BACKGROUND_DATA_PATH, sep='\t')

    # Use and evaluate the model
    rand_samples = samples.sample(n=50, replace=False, random_state=42).reset_index(drop=True)
    background, _ = one_hot_encoding_batch(rand_samples)
    deepShap = DeepShap(model, background)

    # Define the function for scoring sequence attribution
    def score_sequence_attribution_minimal_FIXED(gene, input_miRNA, model, draw_plot=False, step=10, length=50, prediction_threshold=0.5):
        # miRBind takes only 20-long miRNA
        miRNA = input_miRNA[0:20]

        miRNAs = []
        genes = []
        counts = np.zeros(len(gene))

        for i in range(0, len(gene) - length + 1, step):
            start = max(i, 0)
            end = min(i+length, len(gene))
            miRNAs.append(miRNA)
            genes.append(gene[start:end])
            counts[start:end] += 1

        labels = np.zeros(len(genes))

        df = pd.DataFrame({'miRNA': miRNAs, 'gene': genes, 'label': labels})
        data, _ = one_hot_encoding_batch(df, tensor_dim=(50, 20, 1))
        preds = model(data)

        attribution = np.zeros((len(gene), len(miRNA)))
        shap_indices = []
        pred_indices = []
        shap_data = []

        counter = 0
        for i in range(0, len(gene) - length + 1, step):
            if preds[counter][1] > prediction_threshold:
                shap_indices.append(counter)
                pred_indices.append(i)
                shap_data.append(data[counter])
            counter += 1

        if len(shap_data) == 0:
            return []
        shap_data = np.stack(shap_data)

        _, pos_shap = deepShap(shap_data)

        for i in range(len(shap_indices)):
            normalized_shap = pos_shap[i, :, :, 0] * preds[shap_indices[i]][1]

            newrows = np.zeros((pred_indices[i], normalized_shap.shape[1]))
            normalized_shap = np.vstack([newrows, normalized_shap])
            newrows = np.zeros((len(gene) - pred_indices[i] - length, normalized_shap.shape[1]))
            normalized_shap = np.vstack([normalized_shap, newrows])

            attribution += normalized_shap

        attribution = attribution.T.max(axis=0)

        counts[counts == 0] = 1  # Avoid division by zero
        normalized_scores = attribution / counts

        if draw_plot:
            plt.figure(num=random.randint(0, 1000))
            plt.plot(normalized_scores)
            plt.title('Normalized Scores')
            plt.show()

        return normalized_scores

    # Collect results
    miRNA_to_gene_score = {}
    explain_errors = []

    start = time.time()

    i = 0
    for miRNA in mirna_sequences:
        if miRNA not in miRNA_to_gene_score:
            miRNA_to_gene_score[miRNA] = []
        for gene_symbol, gene_sequence in gene_symbol_to_seq.items():
            if isinstance(gene_sequence, str) and len(gene_sequence) > 0:
                try:
                    score = score_sequence_attribution_minimal_FIXED(
                        gene_sequence,
                        miRNA,
                        model,
                        draw_plot=False,
                        step=10,
                        length=50,
                        prediction_threshold=PREDICTION_THRESHOLD
                    )
                    miRNA_to_gene_score[miRNA].append([gene_symbol, score])
                except (AssertionError, ValueError) as e:
                    print(e)
                    explain_errors.append([mirna_name, miRNA, gene_symbol, str(e)])
            else:
                explain_errors.append(gene_symbol)

            i += 1
            if i % 500 == 0:
                print(f"{gene_symbol} | {i} | ", end=" ")

    end = time.time()

    print(f'\nElapsed time: {round(end - start, 2)} seconds ({round((end - start) / 3600, 2)} hours)')

    empties = sum(1 for gene_n_score in miRNA_to_gene_score[my_miRNA] if len(gene_n_score[1]) == 0)
    print(f'{empties} empty scores out of {len(miRNA_to_gene_score[my_miRNA])}')

    # Save the explainability scoring to a file
    with open(SAVE_EXPLAINABILITY_SCORES_PATH, 'w') as file:
        data_to_save = {
            key: [[sub_key, list(sub_val)] for sub_key, sub_val in miRNA_to_gene_score[key]]
            for key in miRNA_to_gene_score.keys()
        }
        json.dump(data_to_save, file, indent=4)

    with open(SAVE_SCANNING_ERRORS_PATH, 'w') as filehandle:
        json.dump(explain_errors, filehandle)

if __name__ == "__main__":
    main()
