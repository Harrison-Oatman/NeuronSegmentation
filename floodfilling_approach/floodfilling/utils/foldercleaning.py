import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2 as cv
import pandas as pd
from skimage.measure import regionprops
from ..inference import inferencesamples
import json
from dataclasses import dataclass, asdict


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

    for file in ["Map2TauImage.png", "barcodes.csv", "preprocessed.png"]:
        shutil.copy2(src_dir+file, dst_dir+file)

    # get boundary image
    body_image, process_image, process_names, cell_image = get_boundaries(src_dir, min_processes=2)
    np.save(dst_dir + "cell_image", cell_image)

    centroids = {"cell_id": [],
                 "soma_centroid_y": [],
                 "soma_centroid_x": []}

    for region in regionprops(body_image):
        centroids["cell_id"].append(region.label)
        centroids["soma_centroid_y"].append(region.centroid[0])
        centroids["soma_centroid_x"].append(region.centroid[1])

    centroids_df = pd.DataFrame(centroids)
    centroids_df.to_csv(dst_dir+"centroids.csv")


def write_inference_test_source_folder(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    for file in ["preprocessed.png"]:
        shutil.copy2(src_dir+file, dst_dir+file)

    # get boundary image
    body_image, process_image, process_names, cell_image = get_boundaries(src_dir, min_processes=2)
    np.save(dst_dir + "cell_image", cell_image)

    centroids = []

    for region in regionprops(body_image):
        centroids.append(region.centroid)

    np.save(dst_dir+"centroids.npy", centroids)

    inference_example = inferencesamples.InferenceSample(
        input=dst_dir+"preprocessed.png",
        centers=dst_dir+"centroids.npy",
        source=src_dir.rstrip("\\")[-4:],
        label=dst_dir+"cell_image.npy"
    )

    json_str = json.dumps(asdict(inference_example), indent=2)

    with open(dst_dir+"inference.json", "w") as f:
        f.write(json_str)


def main(n="0605", train=True):

    src = f"C:\\Lab Work\\segmentation\\training\\{n}\\"

    if train:
        dst = f"C:\\Lab Work\\segmentation\\floodfilling_data\\{n}b\\"
        write_training_source_folder(src, dst)

    else:
        dst = f"C:\\Lab Work\\segmentation\\floodfilling_data\\inference\\"
        write_inference_test_source_folder(src, dst)

