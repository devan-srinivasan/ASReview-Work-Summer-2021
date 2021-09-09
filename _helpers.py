"""Mimicks asreview plot command, was used to learn h5 file breakdown
and its use in the asreview visualization library"""
import numpy as np
import h5py
from typing import Union
import plotly.graph_objects as px


def get_file(filepath: str) -> h5py.File:
    """This function generates an h5py file from a file"""
    try:
        return h5py.File(filepath)
    except FileNotFoundError:
        print('File not found. Check path is correct: {0}'.format(filepath))


def get_key_data(hf: h5py.File, index: Union[int, str], key: str, arr: bool = True) \
        -> Union[np.array, h5py.Dataset]:
    """This function returns the given dataset for a given key

        Preconditions:
        - key in ['pool_idx', 'proba', 'train_idx']

        set arr to False if you don't want the dataset converted to a numpy array
    """
    if type(index) == int:
        index = str(index)
    if index in hf['results'].keys():
        if key in hf['results'][str(index)].keys():
            res = hf['results'][str(index)].get(key)
            if arr:
                res = np.array(res)
            return res
        else:
            raise KeyError('Invalid key: {0}'.format(key))
    else:
        raise KeyError('Invalid index: {0}'.format(index))


def get_ie_data(arr: np.array, inc_arr: np.array, include: bool = True) -> list:
    """
    This function takes an array of record ids and filters them to included/excluded
    """
    return list(filter(lambda elm: inc_arr[elm] == include, arr))


def plot_inclusion(hf: h5py.File, included_arr: np.array, title: str) -> None:
    """
    This function plots from percent indexes
    """
    x, y = [], []
    print("Making Points...", end='')
    idxs = list(sorted(map(int, hf['results'].keys())))
    max_val = idxs[-1]
    for idx in idxs:
        if idx == 0:
            continue
        included = get_ie_data(get_key_data(hf, idx, 'train_idx'), included_arr)
        x.append((idx / max_val) * 100)
        y.append((len(included) / sum(included_arr)) * 100)
    print("Done\n\nOpening in browser")
    fig = px.Figure(data=px.Scatter(x=x, y=y))
    fig.update_layout(title={'text': title})
    fig.show()
