import branches
import metrics
import plotting
import preprocess
import segmentation
import thresholding
import file
import basicrna
import search
import trees
import assignment
import transcriptomics
import boundaries
import colors

import cProfile

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import skimage.morphology as morphology
from skimage.measure import label

from collections import Counter
from tqdm import tqdm

parent_directory = os.path.split(file.current_directory)[0]
datapath = parent_directory + "\\training\\0520\\"
datapath_two = parent_directory + "\\training\\0605\\"
plotpath = parent_directory + "\\plots\\"
plotting.set_dir(plotpath)

# RNA = pd.read_csv(datapath + "barcodes.csv")
RNA = pd.read_csv(datapath + "barcodes_p5_cellCenterDistance.csv")
RNA_two = pd.read_csv(datapath_two + "barcodes.csv")


def save(img,name,dir):
    # saveim = Image.fromarray(img)
    # saveim.save(dir+name)
    np.save(dir+name+".npy",img)


processImageFile = datapath + "processImage.npy"
somaImageFile = datapath + "somaImage.npy"

try:
    print("loading process, soma images")
    processImage = np.load(processImageFile)
    somaImage = np.load(somaImageFile)
    thresholdedProcessImage = np.load(datapath + "thresholdedProcessImage.npy")
    brokenProcessImage = np.load(datapath + "brokenProcessImage.npy")
    cleanedProcessImage = np.load(datapath + "cleanedProcessImage.npy")
    thresholdedSomaImage = np.load(datapath + "thresholdedSomaImage.npy")

    newThreshold = False

except OSError:
    print("some images not found, rebuilding")
    imname = 'preprocessed_Probabilities.png'

    datafile = datapath + imname
    segProbIm = cv2.imread(datafile)

    print("thresholding image")
    thresholdedProcessImage, thresholdedSomaImage = thresholding.threshold_img(segProbIm)
    print("breaking down image")
    brokenProcessImage, togetherskel = segmentation.break_down(thresholdedProcessImage)

    cleanedProcessImage = morphology.remove_small_objects(brokenProcessImage>0,min_size=10,connectivity=1)
    cleanedSomaImage = morphology.remove_small_objects(thresholdedSomaImage>0,min_size=25,connectivity=1)

    processImage = label(cleanedProcessImage,connectivity=1)
    somaImage = label(cleanedSomaImage,connectivity=2)

    save(thresholdedProcessImage, "thresholdedProcessImage", datapath)
    save(brokenProcessImage, "brokenProcessImage", datapath)
    save(cleanedProcessImage, "cleanedProcessImage", datapath)
    save(thresholdedSomaImage, "thresholdedSomaImage", datapath)
    save(processImage, "processImage", datapath)
    save(somaImage, "somaImage", datapath)

    newThreshold = True


# load assignments
try:
    assignments = np.load(datapath+"assignments.npy")
    process_labels = np.load(datapath+"processLabels.npy")
    cell_image = np.load(datapath+"cellImage.npy")
    new_rna = pd.read_csv(datapath+"algRNA.csv")

    # reassign if threshold has just been run
    if newThreshold:
        raise OSError()

except OSError:
    print("running assignment algorithm")
    assignments, process_labels, cell_image, new_rna = assignment.start_to_end(processImage, somaImage, RNA)
    cellImage = np.array(cell_image,dtype=np.int)
    permuted_cell_image = plotting.permute_image(cellImage)
    np.save(datapath+"assignments.npy", assignments)
    np.save(datapath+"processLabels.npy", process_labels)
    np.save(datapath+"cellImage.npy", cellImage)
    new_rna.to_csv(datapath+"algRNA.csv")

print("loading complete, calculating accuracy")

ab_a, ab_b, a_to_b, b_to_a = metrics.align_rna(RNA, new_rna)
alg_master_accuracy = metrics.accuracy_set(ab_a, ab_b)

print("getting human labeled boundaries")

humanBodyImage, humanProcessImage, processNames = boundaries.getBoundaries(datapath)

preprocessed = np.array(cv2.imread(datapath+"Preprocessed.png"))
preprocessed = np.dstack((preprocessed[:, :, 2], preprocessed[:, :, :2]))
preprocessed[:, :, 1] //= 2
ilastik = np.array(cv2.imread(datapath+"preprocessed_Probabilities.png"))
ilastik[:, :, 0] //= 2

bbox = [[700, 700, 500, 500], [2600, 2700, 600, 600]]
# plotting.embedimg(preprocessed, bbox, dir=plotpath, name="preprocessed_embed")
two_map = {0 : '#EDF3FA',
        1 : '#2F5061',
        2 : '#F28066',
        3 : '#5EA2C4',
        4 : '#FFAA96',
        }

two_map = {0 : '#F5FAFF',
        1 : '#2F5061',
        2 : '#F28066',
        3 : '#2790C4',
        4 : '#FF7859',
        }

diverge = {0 : '#EDF3FA',
        1 : '#698EBF',
        2 : '#5CBF84',
        3 : '#E0EB9D',
        4 : '#FFAA96',
        }

two_blue = {0 : '#F5FAFF',
        2 : '#2F5061',
        1 : '#2790C4',
        }

two_orange = {0 : '#F5FAFF',
        1 : '#FFBAAB',
        2 : '#E35434',
        }

two_map_mapped = colors.seqfromdict(two_map)
diverge_mapped = colors.seqfromdict(diverge)

print("complete")