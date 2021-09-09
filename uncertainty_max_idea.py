"""This script is the application of Alex's idea of optimizing ASReview by combining
the uncertainty and max query strategies to make ELAS smarter.

Ensure:
    DATASET_NAME is a file in Datasets folder
        Dataset file must be csv
    0 < UNCERTAINTY_PERCENTILE < 1
"""
import asreview.models.classifiers
import asreview.models.feature_extraction
import asreview.models.query
import asreview.models.balance
import pandas as pd
import os
from _helper_functions import get_file, get_key_data, get_ie_data, plot_inclusion
from datetime import datetime

# -------------------------------- USER SETS THESE
DATASET_NAME = 'CreativityF.csv'
CLASSIFIER = 'nb'
FEATURE_EXTRACTION = 'tfidf'
BALANCE = 'double'
N_PRIOR_INCLUDED = 5
N_PRIOR_EXCLUDED = 10
UNCERTAINTY_PERCENTILE = 0.09
# --------------------------------

out_unc_name = DATASET_NAME.split('.csv')[0] + '_unc.h5'
out_max_name = DATASET_NAME.split('.csv')[0] + '_max.h5'

df = pd.read_csv(DATASET_NAME)

dir_name = DATASET_NAME.split('.csv')[0] + datetime.now().strftime("%H:%M:%S") + '_unc_max_results'
if dir_name not in os.listdir():
    os.mkdir(dir_name)

asreview.ReviewSimulate(asreview.ASReviewData.from_file('Datasets/' + DATASET_NAME),
                        model=asreview.models.classifiers.get_classifier(CLASSIFIER),
                        feature_model=asreview.models.feature_extraction.get_feature_model(
                            FEATURE_EXTRACTION),
                        balance_model=asreview.models.balance.get_balance_model(
                            BALANCE),
                        query_model=asreview.models.query.get_query_model('uncertainty'),
                        n_prior_included=N_PRIOR_INCLUDED,
                        n_prior_excluded=N_PRIOR_EXCLUDED,
                        state_file=dir_name + '/' + out_unc_name).review()

hf = get_file(dir_name + '/' + out_unc_name)
idx = str(int(UNCERTAINTY_PERCENTILE * max(map(int, hf['results'].keys()))))

train = get_key_data(hf, idx, 'train_idx')
prior_inc = get_ie_data(train, df['Included'], include=True)
prior_exc = get_ie_data(train, df['Included'], include=False)

asreview.ReviewSimulate(asreview.ASReviewData.from_file('Datasets/' + DATASET_NAME),
                        model=asreview.models.classifiers.get_classifier(CLASSIFIER),
                        feature_model=asreview.models.feature_extraction.get_feature_model(
                            FEATURE_EXTRACTION),
                        balance_model=asreview.models.balance.get_balance_model(
                            BALANCE),
                        query_model=asreview.models.query.get_query_model('max'),
                        n_prior_included=N_PRIOR_INCLUDED,
                        n_prior_excluded=N_PRIOR_EXCLUDED,
                        state_file=dir_name + '/' + out_unc_name).review()
