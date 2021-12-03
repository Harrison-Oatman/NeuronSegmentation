import numpy as np
from PIL import Image

def write_soma_channel(image, rna):
    newimage = image.copy()
    somabarcodes = np.load(constpath+"somabarcodes.npy")
    somarna = rna[np.isin(rna["barcode_id"], somabarcodes)]
    xs = np.array(somarna["global_x"]).astype(np.int32)
    ys = np.array(somarna["global_y"]).astype(np.int32)
    print(len(xs))
    newimage[ys,xs,0] = 255
    return newimage

def write_process_channel(image, rna):
    newimage = image.copy()
    newimage[:,:,1] = 0
    processbarcodes = np.load(constpath+"processbarcodes.npy")
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
