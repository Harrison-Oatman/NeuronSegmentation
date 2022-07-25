import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import cv2 as cv
import pandas as pd
from skimage.measure import regionprops
from ..inference import inferencesamples
from .. import const


def get_boundaries(datapath, min_processes=0):

    # get size of immunofluorescence image
    combined_if_image = plt.imread(datapath+"Map2TauImage.png")
    (width, height) = combined_if_image.shape[:2]

    matfile = "modifiedBoundaries.mat"

    # create arrays to story body and process locations
    body_image = np.zeros((width, height), dtype=np.int32)
    process_image = np.zeros((width, height), dtype=np.int32)
    cell_image = np.zeros((width, height), dtype=np.int32)

    # track names to determine parent relationships
    process_names = ["0.0"]

    # import from mat file a single variable
    boundaries = sio.loadmat(datapath+matfile)['boundaries']

    process_count = 0
    body_count = 0

    # loop over cells and plot cell body
    for cc in range(0, boundaries.shape[0]):
        body_count += 1
        # get polygon describing boundaries of soma
        body_boundary = boundaries[cc, 0]['bodyBoundary']

        # import from mat file a single variable
        process_boundary = boundaries[cc, 0]['processBoundary']

        # skip cell
        if body_boundary.shape[0] == 0:
            continue

        if process_boundary.shape[1] < min_processes:
            continue

        # get names of processes
        names = np.array(boundaries[cc,0][3])
        if len(names) > 0:
            names = names[0]

        for bb in range(0, process_boundary.shape[1]):
            process_count += 1
            # fill in polygon boundary of each process
            points = np.array(list(zip(process_boundary[0, bb][:, 0], process_boundary[0, bb][:, 1])), dtype=np.int32)
            cv.fillPoly(process_image, [points], process_count,)
            cv.fillPoly(cell_image, [points], body_count)
            # give each process a unique name
            process_names.append(str(body_count) + "." + str(names[bb]))

        # fill in polygon boundary of each body
        points = np.array(list(zip(body_boundary[:, 0], body_boundary[:, 1])), dtype=np.int32)
        cv.fillPoly(body_image, [points], body_count)
        cv.fillPoly(cell_image, [points], body_count)

    return body_image, process_image, process_names, cell_image


def write_training_source_folder(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    # for file in ["Map2TauImage.png", "barcodes.csv", "preprocessed.png"]:
    for file in ["Map2TauImage.png", "barcodes.csv"]:  # rm preprocessed
        shutil.copy2(src_dir+file, dst_dir+file)

    # get boundary image
    min_processes = const.MIN_PROCESSES
    body_image, process_image, process_names, cell_image = get_boundaries(src_dir, min_processes)
    np.save(dst_dir + "cell_image", cell_image)
    np.save(dst_dir + "process_image", process_image)
    np.save(dst_dir + "process_names", process_names)
    np.save(dst_dir + "body_image", body_image)
    #
    # # find seed points
    # erosion = const.EROSION
    # centroids = find_seed_pts(cell_image, erosion)
    # centroids.to_csv(dst_dir + "centroids.csv")


def write_inference_test_source_folder(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    for file in ["barcodes.csv", "Map2TauImage.png"]:
        shutil.copy2(src_dir+file, dst_dir+file)

    # get boundary image
    min_processes = const.MIN_PROCESSES
    body_image, process_image, process_names, cell_image = get_boundaries(src_dir, min_processes)

    np.save(dst_dir + "cell_image", cell_image)
    np.save(dst_dir + "process_image", process_image)
    np.save(dst_dir + "process_names", process_names)
    np.save(dst_dir + "body_image", body_image)


def main(n="0605", train=True):

    src = f"C:\\Lab Work\\segmentation\\training\\{n}\\"

    if train:
        dst = f"C:\\Lab Work\\segmentation\\floodfilling_data\\{n}\\"
        write_training_source_folder(src, dst)

    else:
        dst = f"C:\\Lab Work\\segmentation\\floodfilling_data\\inference\\{n}\\"
        write_inference_test_source_folder(src, dst)

