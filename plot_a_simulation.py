"""
This script plots the result of a simulation. Graph will be opened in your browser
"""
import pandas as pd
from _helpers import get_file, plot_inclusion

# -------------------------------- USER SETS THESE
DATASET_PATH = 'Datasets/CreativityF.csv'
H5_FILE_PATH = 'h5_results/test02.h5'
PLOT_TITLE = 'your_choice'
# --------------------------------

plot_inclusion(get_file(H5_FILE_PATH),
               pd.read_csv(DATASET_PATH)['Included'], PLOT_TITLE)
