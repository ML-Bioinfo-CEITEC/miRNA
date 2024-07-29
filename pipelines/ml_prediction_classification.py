import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score

from funmirtar.models.constants import SEEDS_TO_COUNT, SEED_COUNT_COLUMNS, GLOBAL_FEATURES, LOCAL_FEATURES, CLASSIFICATION_COLUMNS
from funmirtar.utils.plots import plot_feature_importance
from funmirtar.utils.file import make_dir_with_parents, extend_path_by_suffix_before_filetype, extract_file_name_from_path, insert_text_before_the_end_of_path

def main():
    parser = argparse.ArgumentParser(description='Run model training and output model predictions for the test data.')
    parser.add_argument(
        '--train_file_path',
        type=str, 
        default= "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl",
        help='Path to the train dataset file.'
    )
    parser.add_argument(
        '--test_file_path',
        type=str, 
        default= "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl",
        help='Path to the test dataset file.'
    )
    parser.add_argument(
        '--run_name',
        type=str, 
        default= 'seeds.signal.local_features',
        help='Name of the run. The resulting folder and prediction columns of the resulting dataframe will be named this way.'
    )
    parser.add_argument(
        '--out_folder_path', 
        type=str, 
        default="../data/predictions/",
        help='Path to the folder that will containe the folder with the results. The final location will be out_folder_path + run_name.'
    )
    
    # setting the parameters
    args = parser.parse_args()
    
    RUN_NAME = args.run_name    
    
    IN_FEATURES_PATH_TRAIN = args.train_file_path
    IN_FEATURES_PATH_TEST = args.test_file_path

    OUT_FOLDER_PATH = f"{args.out_folder_path}{RUN_NAME}/"
    PREDICTION_TYPE = '.class_preds'
    FILE_NAME_TRAIN = extract_file_name_from_path(args.train_file_path)
    FILE_NAME_TEST = extract_file_name_from_path(args.test_file_path)

    OUT_PATH_TRAIN = Path(
        OUT_FOLDER_PATH + insert_text_before_the_end_of_path(FILE_NAME_TRAIN, PREDICTION_TYPE)
    )
    OUT_PATH_TEST = Path(
        OUT_FOLDER_PATH + insert_text_before_the_end_of_path(FILE_NAME_TEST, PREDICTION_TYPE)
    )
    
    COLUMNS_FOR_PRED = []
    COLUMNS_FOR_PRED.extend(GLOBAL_FEATURES)
    COLUMNS_FOR_PRED.extend(LOCAL_FEATURES)
    COLUMNS_FOR_PRED.extend(SEEDS_TO_COUNT)
    COLUMNS_FOR_PRED.extend(SEED_COUNT_COLUMNS)
    
    # TODO make the model list an argument of the script
    MODEL_LIST = [
        f'logistic_regression.{RUN_NAME}',
        f'gradient_boosting_classifier.{RUN_NAME}',
        f'xgb.{RUN_NAME}',
        f'random_forest.{RUN_NAME}',
    ]
    # MODEL_LIST = [
    #     f'random_forest.default.{RUN_NAME}',
    #     f'random_forest.optimised.{RUN_NAME}',
    # ]
    OUT_COLUMNS = []
    OUT_COLUMNS.extend(CLASSIFICATION_COLUMNS)
    OUT_COLUMNS.extend(MODEL_LIST)
    
    make_dir_with_parents(OUT_FOLDER_PATH)

    data_train = pd.read_pickle(IN_FEATURES_PATH_TRAIN)
    data_test = pd.read_pickle(IN_FEATURES_PATH_TEST)

    utils.check_random_state(3)
    np.random.seed(1)

    
    # training the models and prediction

    # HOTFIX until we regenerate new data file with corrected naming
    data_train.rename(columns={'kmer6_mismatch_count':'kmer6_bulge_or_mismatch_count'}, inplace=True)
    data_test.rename(columns={'kmer6_mismatch_count':'kmer6_bulge_or_mismatch_count'}, inplace=True)

    x_train = data_train[COLUMNS_FOR_PRED].fillna(0,inplace=False)
    x_test = data_test[COLUMNS_FOR_PRED].fillna(0,inplace=False)

    y_train = data_train.label
    y_test = data_test.label

    """### Train models

    #### Logistic regression
    """

    model_lr = LogisticRegression(max_iter=10000)
    model_lr.fit(x_train, y_train)

    y_pred_lr_train = model_lr.predict_proba(x_train)
    y_pred_lr_test = model_lr.predict_proba(x_test)

    data_train[f'logistic_regression.{RUN_NAME}']=y_pred_lr_train[:,1]
    data_test[f'logistic_regression.{RUN_NAME}']=y_pred_lr_test[:,1]

    """#### (Histogram) Gradient Boosting Classifier"""

    model_grad = GradientBoostingClassifier()
    model_grad.fit(x_train, y_train)

    y_pred_grad_train = model_grad.predict_proba(x_train)
    y_pred_grad_test = model_grad.predict_proba(x_test)

    data_train[f'gradient_boosting_classifier.{RUN_NAME}'] = y_pred_grad_train[:,1]
    data_test[f'gradient_boosting_classifier.{RUN_NAME}'] = y_pred_grad_test[:,1]

    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_classifier.fit(x_train, y_train)

    # Make predictions with XGBOOST
    xgb_y_pred_class = xgb_classifier.predict(x_test)
    xgb_y_pred_test = xgb_classifier.predict_proba(x_test)

    xgb_y_pred_train = xgb_classifier.predict_proba(x_train)

    # Evaluate the XGBOOST classifier
    xgb_accuracy = accuracy_score(y_test, xgb_y_pred_class)
    xgb_report = classification_report(y_test, xgb_y_pred_class)

    # print(f'Accuracy (XGBoost): {xgb_accuracy}')
    # print('Classification Report (XGBoost):')
    # print(xgb_report)

    data_test[f'xgb.{RUN_NAME}'] = xgb_y_pred_test[:,1]
    data_train[f'xgb.{RUN_NAME}'] = xgb_y_pred_train[:,1]

    # best 250, 20, 20, 10, sqrt
    # 250, 20, 100, 40, sqrt
    # 100, None, 200, 40, 0.5,
    rf_classifier = RandomForestClassifier(
        n_estimators=250, # 100, 150, 200, 250
        max_depth=20, # None, 10, 20
        min_samples_split=100,  # 2, 5, 10, 20, 100, 150
        min_samples_leaf=40, # 1, 2, 4, 6, 10, 40, 60
        max_features='sqrt', # 'auto', 'sqrt', 'log2', float
        random_state=42,
    )
    rf_classifier.fit(x_train, y_train)

    # Make predictions
    rf_y_pred_class = rf_classifier.predict(x_test)
    rf_y_pred = rf_classifier.predict_proba(x_test)

    rf_y_pred_train = rf_classifier.predict_proba(x_train)

    # Evaluate the classifier
    rf_accuracy = accuracy_score(y_test, rf_y_pred_class)
    rf_report = classification_report(y_test, rf_y_pred_class)

    # print(f'Accuracy (Random Forest): {rf_accuracy}')
    # print('Classification Report (Random Forest):')
    # print(rf_report)
    
    
    data_test[f'random_forest.{RUN_NAME}'] = rf_y_pred[:,1]
    data_train[f'random_forest.{RUN_NAME}'] = rf_y_pred_train[:,1]

    feature_importances = rf_classifier.feature_importances_
    feature_names = x_train.columns
    plot_feature_importance(feature_names, feature_importances, 40, OUT_FOLDER_PATH + 'features')

    data_train[OUT_COLUMNS].to_pickle(OUT_PATH_TRAIN)
    data_test[OUT_COLUMNS].to_pickle(OUT_PATH_TEST)

    
if __name__ == "__main__":
    main()


