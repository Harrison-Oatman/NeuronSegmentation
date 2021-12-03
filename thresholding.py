from skimage.morphology import dilation, erosion, closing, opening, disk

"""
Dendrite Extraction
"""
def threshold_dendrites(im):

    hi = 0.7
    lo = 0.25
    closerad = 8
    dilrad = 5
    lorad = 1
    minsize = 250
    holeMax = 2000
    holeMin = 20

    # generate disk filters
    closefilter = disk(closerad)
    dilfilter = disk(dilrad)
    lofilter = disk(lorad)

    # threshold dendrites with high and low thresholds
    hithresh,hiThresholdedDendrite=cv2.threshold(im,255*hi,255,cv2.THRESH_BINARY)
    lothresh,loThresholdedDendrite=cv2.threshold(im,255*lo,255,cv2.THRESH_BINARY)

    hibar = np.zeros(hiThresholdedDendrite.shape,dtype=np.int)
    lobar = np.zeros(loThresholdedDendrite.shape,dtype=np.int)

    hibar[hiThresholdedDendrite > 0] = 1
    lobar[loThresholdedDendrite > 0] = 1

    # find large holes in high threshold to preserve true holes
    holemask = morphology.remove_small_holes(hibar>0, holeMax) - hibar
    holemask = morphology.remove_small_objects(holemask>0, holeMin)

    # remove debris from low threshold
    lobar = morphology.remove_small_objects(lobar>0, min_size=minsize, connectivity=1)

    # close hi threshold to bridge gaps
    hibar = closing(hibar, closefilter)

    # smaller dilation to high threshold
    hibar = dilation(hibar, dilfilter)

    g = np.bitwise_and(hibar>0,lobar>0)
    g[holemask > 0] = 0

    ##try to remove pieces of dendrites that are too round to be dendrites
    #calculate circularity of objects
    label_dendrite_img = measure.label(g, connectivity=2)
    props = measure.regionprops(label_dendrite_img)
    allPerimeters = [prop.perimeter for prop in props]
    allAreas = [prop.area for prop in props]
    labels = [prop.label for prop in props] ##note!! properties are not saved in order as the original labeled image.
    allCircularities = np.array(allPerimeters)**2 / (4 * np.pi* np.array(allAreas))

    #"color" objects based on their allCircularities value
    circularity_dendrite_img=np.zeros(label_dendrite_img.shape)

    for idx,label in enumerate(labels):
        circularity_dendrite_img[label_dendrite_img == label] = allCircularities[idx]


    #remove objects that are about circular
    circ_threshold=5
    g[circularity_dendrite_img<circ_threshold]=0

    # fill in small holes
    g = morphology.remove_small_holes(g, holeMin)

    # ensure datatype
    g = np.array(g, dtype=np.uint8)

    return g

"""
Soma Extraction
"""

def threshold_soma(im):

    hi = 0.9
    lo = 0.1
    dilrad = 8

    dilfilter = disk(dilrad)

    # find high threshold and low threshold
    somathresh, hiThresholdedSoma=cv2.threshold(im,255*hi,255,cv2.THRESH_BINARY)
    somathresh, loThresholdedSoma=cv2.threshold(im,255*lo,255,cv2.THRESH_BINARY)

    hiso = np.zeros(im.shape)
    loso = np.zeros(im.shape)

    hiso[hiThresholdedSoma > 0] = 1
    loso[loThresholdedSoma > 0] = 1

    # dilate high threshold
    hiso = dilation(hiso, dilfilter)

    # final segmentation is intersection
    b = np.bitwise_and(hiso>0,loso>0)

    return b

"""
Thresholding function
"""

def threshold_img(im):
    r = np.zeros(im[:,:,1].shape)
    g = threshold_dendrites(im[:,:,1])
    b = threshold_soma(im[:,:,2])

    g[b > 0] = 0

    testImage = np.dstack((r,g,b))

    # embedplot(testImage, bbs, name=f"union",show=True)

    return g, b
