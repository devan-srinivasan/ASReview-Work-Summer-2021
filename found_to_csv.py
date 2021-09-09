"""This file is used for finding lost papers from an ASReview simulation. It serves as a
way to QUANTIFY the error of a simulation. It furthermore writes this result to a csv file
that can be viewed in excel for easy analysis.

To use this script ensure the following:
    The h5 file you are reading from is in the folder 'h5_results'
    The dataset you are reading from is in the folder 'Datasets'
        Datasets must be csv format
"""
import pandas as pd
import h5py
from _helpers import get_key_data, get_ie_data

# -------------------------------- USER SETS THESE
H5_FILE_NAME = 'test02.h5'
DATASET_NAME = 'CreativityF.csv'
PERCENTILE_MARKS = [80, 90]
# --------------------------------

df = pd.read_csv('Datasets/' + DATASET_NAME)
hf = h5py.File('h5_results/' + H5_FILE_NAME)

percents = sorted(PERCENTILE_MARKS)
idxs = [0] * len(percents)
founds = [[]] * len(percents)
for key in range(1, int(max(map(int, hf['results'].keys())))):
    found = get_ie_data(get_key_data(hf, key, 'train_idx'), df['Included'])
    for i in range(len(percents)):
        if not idxs[i] and abs((len(found) / sum(df['Included'])) - (percents[i] / 100)) < 0.001:
            idxs[i] = key
            founds[i] = found
            break

for j, idx in enumerate(idxs):
    out = pd.DataFrame(columns=df.columns[0:5])
    for id in founds[j]:
        out = out.append(dict(df.iloc[id][0:5]), ignore_index=True)

    out_name = H5_FILE_NAME.split('.h5')[0]
    out.to_csv('csv_result_files/' + out_name + '_found_' + str(percents[idxs.index(idx)]) + '.csv',
               index=False)
print('DONE...check csv_result_files for results')
