import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2 as cv

def getBoundaries(datapath):

    #get size of immunofluorescence image
    combinedIFimage = plt.imread(datapath+"Map2TauImage.png")
    (width, height)=combinedIFimage.shape[:2]

    matfile = "modifiedBoundaries.mat"

    # create arrays to story body and process locations
    bodyImage = np.zeros((width, height),dtype=np.int32)
    processImage = np.zeros((width, height),dtype=np.int32)

    # track names to determine parent relationships
    processNames = ["0.0"]

    #import from mat file a single variable
    boundaries= sio.loadmat(datapath+matfile)['boundaries']

    processCount = 0
    bodyCount = 0

    #loop over cells and plot cell body
    for cc in range(0,boundaries.shape[0]):
        bodyCount += 1
        #get polygon describing boundaries of soma
        bodyBoundary = boundaries[cc,0]['bodyBoundary']

        #import from mat file a single variable
        processBoundary = boundaries[cc,0]['processBoundary']

        #skip cell
        if bodyBoundary.shape[0]==0:
            continue

        # get names of processes
        names = np.array(boundaries[cc,0][3])
        if len(names) > 0:
            names = names[0]

        for bb in range(0,processBoundary.shape[1]):
            processCount += 1
            # fill in polygon boundary of each process
            points = np.array(list(zip(processBoundary[0,bb][:,0],processBoundary[0,bb][:,1])),dtype=np.int32)
            cv.fillPoly(processImage, [points], processCount,)
            # give each process a unique name
            processNames.append(str(bodyCount) + "." + str(names[bb]))

        # fill in polygon boundary of each body
        points = np.array(list(zip(bodyBoundary[:,0],bodyBoundary[:,1])),dtype=np.int32)
        cv.fillPoly(bodyImage, [points], bodyCount)


    return bodyImage, processImage, processNames