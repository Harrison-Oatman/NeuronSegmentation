from ..utils.cropping import imalign, crop
from ..const import *
import numpy as np
import matplotlib.pyplot as plt


def example_crop(example, example_data, data):
    x, y, = int(example["centerx"]), int(example["centery"])
    a_slice, b_slice = imalign(data, example_data["pom"], (y, x))
    # print("slices", a_slice, b_slice)
    return data[a_slice]


def pair_crop(pair, data):
    center = pair[0][:-1]
    crop_shape = [BRANCH_WINDOW_SIZE, BRANCH_WINDOW_SIZE]
    # print(data.shape)
    return crop(data, center, crop_shape)


def pair_label(pair, label, branch_im):
    mla, mlb = pair[1][0], pair[1][1]  # machine label a

    if label is None:
        return None

    hlsa = label[branch_im == mla]
    hlsb = label[branch_im == mlb]

    if len(hlsa) < MIN_BRANCH_LEN or len(hlsb) < MIN_BRANCH_LEN:
        return None

    values, counts = np.unique(hlsa, return_counts=True)
    ind = np.argmax(counts)
    hla = values[ind]

    values, counts = np.unique(hlsb, return_counts=True)
    ind = np.argmax(counts)
    hlb = values[ind]

    if hla == 0 or hlb == 0:
        return None

    return hlb == hla


def separate_data(pair, pom: np.ndarray, branch_seg):
    mla, mlb = pair[1]
    bs = branch_seg
    channel1 = (bs == mla) * pom.copy()
    channel2 = (bs == mlb) * pom.copy()
    channel3 = np.bitwise_and(bs != mla, bs != mlb) * pom.copy()

    return np.dstack((channel1, channel2, channel3))





