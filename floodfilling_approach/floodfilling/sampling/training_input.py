import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


def simple_inputs(if_image=None, rna=None, **kwargs):
    """
    if_image: [y, x, 3] channels [map2, tau, ---]
    rna: df with 'global_y, global_x'

    input_image: [y, x, 2] channels [map2, rna_counts]
    """
    rna_image = rna_counts(rna, if_image[:, :, 0])
    input_image = np.dstack([if_image[:, :, 2:3], rna_image])
    return input_image


def rna_counts(rna: DataFrame, arr):

    arr = np.zeros(arr.shape)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, -1)

    x = rna["global_x"].array.to_numpy()
    y = rna["global_y"].array.to_numpy()

    indx = np.array(np.floor(x), dtype=np.int)
    indy = np.array(np.floor(y), dtype=np.int)

    for y, x in zip(indy, indx):
        arr[y, x] += 1

    # plt.imshow(arr)
    # plt.show()

    return arr
