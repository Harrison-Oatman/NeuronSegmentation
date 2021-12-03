import pandas as pd
import numpy as np
import cv2
import skimage.morphology as morphology
from skimage.measure import label
import os

from thresholding import threshold_img
from segmentation import  break_down
from branches import buildSomas, buildProcesses
from plotting import implot

def save(img,name,dir):
    # saveim = Image.fromarray(img)
    # saveim.save(dir+name)
    np.save(dir+name+".npy",img)

def main(datapath):
    RNA = pd.read_csv(datapath + "barcodes.csv")

    processImageFile = datapath + "processImage.npy"
    somaImageFile = datapath + "somaImage.npy"

    if not (os.path.exists(processImageFile) and os.path.exists(somaImageFile)):

        imname = 'preprocessed_Probabilities.png'

        datafile = datapath + imname
        segProbIm = cv2.imread(datafile)

        print("thresholding image")
        thresholdedProcessImage, somaImage = threshold_img(segProbIm)
        print("breaking down image")
        brokenProcessImage, togetherskel = break_down(thresholdedProcessImage)

        cleanedProcessImage = morphology.remove_small_objects(brokenProcessImage>0,min_size=10,connectivity=1)

        processImage = label(cleanedProcessImage,connectivity=1)
        somaImage = label(somaImage,connectivity=1)

        save(processImage, "processImage", datapath)
        save(somaImage, "somaImage", datapath)

    processImage = np.load(processImageFile)
    somaImage = np.load(somaImageFile)

    processes = buildProcesses(processImage, RNA)
    somas = buildSomas(somaImage, RNA)

    implot(processImage, datapath)






if __name__ == '__main__':
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0]
    datapath = parent_directory + "\\training\\0520\\"
    main(datapath)
