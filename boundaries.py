import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2 as cv

def getBoundaries(datapath):

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
