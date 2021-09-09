"""This file is used to run an ASReview simulation.
Simply set the parameters in the parameters file then run this file to
undergo the simulation

Ensure:
    DATASET_NAME is a file in Datasets folder
        Dataset file must be csv
"""
import asreview.models.classifiers
import asreview.models.feature_extraction
import asreview.models.query
import asreview.models.balance

# -------------------------------- USER SETS THESE
DATASET_NAME = 'CreativityF.csv'
CLASSIFIER = 'svm'
QUERY = 'uncertainty'
FEATURE_EXTRACTION = 'doc2vec'
BALANCE = 'double'
NAME = 'test02'
N_PRIOR_INCLUDED = 5
N_PRIOR_EXCLUDED = 10
# --------------------------------

asreview.ReviewSimulate(asreview.ASReviewData.from_file('Datasets/' + DATASET_NAME),
                        model=asreview.models.classifiers.get_classifier(CLASSIFIER),
                        feature_model=asreview.models.feature_extraction.get_feature_model(
                            FEATURE_EXTRACTION),
                        balance_model=asreview.models.balance.get_balance_model(
                            BALANCE),
                        query_model=asreview.models.query.get_query_model(QUERY),
                        n_prior_included=N_PRIOR_INCLUDED,
                        n_prior_excluded=N_PRIOR_EXCLUDED,
                        state_file='h5_results/' + NAME + '.h5').review()
print('DONE...check h5_results folder for h5 result')
