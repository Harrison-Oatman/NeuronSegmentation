import numpy as np
import pandas as pd
import cv2
from PIL import Image


def write_soma_channel(image, rna):
    newimage = image.copy()
    somabarcodes = np.load("const\\"+"somabarcodes.npy")
    somarna = rna[np.isin(rna["barcode_id"], somabarcodes)]
    xs = np.array(somarna["global_x"]).astype(np.int32)
    ys = np.array(somarna["global_y"]).astype(np.int32)
    print(len(xs))
    newimage[ys,xs,0] = 255
    return newimage

def write_process_channel(image, rna):
    newimage = image.copy()
    newimage[:,:,1] = 0
    processbarcodes = np.load("const\\"+"processbarcodes.npy")
    processrna = rna[np.isin(rna["barcode_id"], processbarcodes)]
    xs = np.array(processrna["global_x"]).astype(np.int32)
    ys = np.array(processrna["global_y"]).astype(np.int32)
    print(len(xs))
    newimage[ys,xs,1] = 255
    return newimage

def preprocess(imgfile, rnafile, out):
    rna = pd.read_csv(rnafile)
    img = np.array(cv2.imread(imgfile))
    img = write_soma_channel(img, rna)
    img = write_process_channel(img, rna)
    saveim = Image.fromarray(img)
    saveim.save(out)
    return img

def small_preprocess(imgfile, rnafile, out):
    rna = pd.read_csv(rnafile)
    img = np.array(cv2.imread(imgfile))
    img = write_soma_channel(img, rna)
    img = write_process_channel(img, rna)
    saveim = Image.fromarray(img[:1000,:1000])
    saveim.save(out)
    return img

def small_preprocess(imgfile, rnafile, out):
    rna = pd.read_csv(rnafile)
    img = np.array(cv2.imread(imgfile))
    img = write_soma_channel(img, rna)
    img = write_process_channel(img, rna)
    saveim = Image.fromarray(img[500:1000,500:1000])
    saveim.save(out)
    return img

def preprocessing_stage(datapath):
    imgfile = datapath + "Map2TauImage.png"
    rnafile = datapath + "barcodes.csv"
    outfile = datapath + "small_preprocessed.png"
    preprocessed = small_preprocess(imgfile, rnafile, outfile)
    return preprocessed

if __name__ == '__main__':
    import os
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0]
    datapath = parent_directory + "\\training\\0605\\"
    preprocessing_stage(datapath)
