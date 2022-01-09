import numpy as np
import matplotlib.pyplot as plt
import plotting
from tqdm import tqdm

import scipy.spatial as spatial
from scipy.spatial.distance import cdist
from skimage.feature import canny
from skimage.measure import regionprops
import heapq


class Segment:
    """
    Class to represent a region of dendrite tissue which can contain RNA
    data.
    """
    def __init__(self,props):
        """
        At initialization, we store some spatial properties of the region
        """
        self.nrna = None
        self.rna_index = None
        self.rna_px = None
        self.rna_loc = None
        self.label = props.label
        self.neighbors = {}

        # get the location of the segment within the greater image
        self.ymin, self.xmin, self.ymax, self.xmax = props.bbox
        self.y, self.x = props.centroid

        # get the location of the segment in the local image
        regionys, regionxs = np.where(props.image)
        self.xs = regionxs
        self.ys = regionys

        # region object
        self.region = Region(props.image)

    def addRNA(self, iscontained, rnaloc, rnabc):
        """
        Adds rna data to the segment

        ---Arguments---
        iscontained: truth ndarray of which RNA in image are contained in Segment
        rnaloc: ndarray of y,x locations of RNA
        rnabc: ndarray of rna barcodes
        """
        # handles the rna contained within the segment
        self.rna_loc = rnaloc[iscontained]
        self.rna_px = np.array(np.floor(self.rna_loc), dtype=np.int)
        self.rna_index = np.arange(rnaloc.shape[0],dtype=np.int)[iscontained]

        # self.rna_bc = rnabc[iscontained]
        self.nrna = self.rna_loc.shape[0]

    def getNeighbors(self, labeledImage, label, max_dist, exclude=None):
        """
        Finds all labels contained within a distance from the region

        ---Arguments---
        labeledImage: the image to check
        label: specifies label to store the neighbor list (e.g. "soma")
        max_dist: how far to search for neighbors
        exclude: label to exclude, does not exclude if None

        ---Returns---
        self.neighbors[label]: list of neighbors found
        """
        n, m = labeledImage.shape
        neighbors = []
        d = max_dist

        # trim to keep smaller image within larger image
        ymin = max(0,self.ymin-d)
        ymax = min(n-1,self.ymax+d-1)
        xmin = max(0,self.xmin-d)
        xmax = min(m-1,self.xmax+d-1)

        # find other objects within smaller image
        nbs = np.unique(labeledImage[ymin:ymax,xmin:xmax])

        # remove background from neighbor list
        nbs = nbs[nbs!=0]

        # remove self from neighbor list
        if exclude is not None:
            nbs = nbs[nbs!=exclude]

        self.neighbors[label] = nbs

        return self.neighbors[label]

    def distance_from(self, segment):
        """
        Calculates the minimum distance between self and segment

        ---Arguments---
        segment: segment to compare

        ---Returns---
        aclosest: closest pixel in self (self coordinates)
        bclosest: closest pixel in segment (segment coordinates)
        dist.min(): distance between pixels
        """
        a = np.array(np.where(self.region.edges))
        a[0] += self.ymin
        a[1] += self.xmin
        a = a.T
        b = np.array(np.where(segment.region.edges))
        b[0] += segment.ymin
        b[1] += segment.xmin
        b = b.T
        dist = cdist(a,b)
        k,m = np.unravel_index(np.argmin(dist),dist.shape)

        aclosest = (a[k][0] - self.ymin, a[k][1] - self.xmin)
        bclosest = (b[m][0] - segment.ymin, b[m][1] - segment.xmin)

        return aclosest, bclosest, dist.min()


class Process(Segment):
    """
    Subclass used for dendrites
    """
    def get_rnadistances(self, source):
        """
        Get the distance of each RNA in the region from the source pixel
        Calculates distance along the segment, not euclidean distance

        ---Arguments---
        source: pixel within segment to check

        ---Returns---
        list of distances of each RNA in segment
        """
        # using region properties get the distance of each pixel inthe RNA
        disgraph = self.region.make_disgraph(source)
        self.dis = disgraph
        return disgraph[self.rna_px[:,0]-self.ymin,self.rna_px[:,1]-self.xmin]


class Soma(Segment):
    pass


class Region(object):
    """
    A region of pixels within an image
    """
    def __init__(self, binaryimage):
        """
        Uses binaryimage from regionprops
        """
        self.im = binaryimage
        self.ymax,self.xmax = self.im.shape
        self.find_edges()

    def make_disgraph(self, pt):
        """
        calculate distance along region from pt, takes distance traveling
        along the contained pixels

        ---Arguments---
        pt: the pixel to calculate distance from. Must be contained in the region

        ---Returns---
        disIm: self.im with values corresponding to distance
                - (pixels outside or not reached get -5)
        """
        # use search algorithm to determine shortest paths to a point
        disIm = (1 - self.im)*np.inf
        searched = 1 - self.im
        disIm = np.pad(disIm, 1, constant_values=np.inf)
        disIm = np.array(disIm, dtype=np.float)
        searched = np.pad(searched, 1, constant_values=1)
        mincost = np.zeros(disIm.shape)
        mincost[:,:] = np.inf

        searched[pt[0]+1,pt[1]+1] = 1

        candidates = self.neighbors((pt[0] + 1,pt[1] + 1),0)
        candidates = [candidate for candidate in candidates if searched[candidate[1][0],candidate[1][1]] == 0]
        frontier = []
        heapq.heapify(frontier)
        for candidate in candidates:
            heapq.heappush(frontier,candidate)

        while len(frontier) > 0:
            # loop, add current point to
            currentpt = heapq.heappop(frontier)
            val = currentpt[0]
            y = currentpt[1][0]
            x = currentpt[1][1]

            if searched[y,x] == 1:
                continue
            else:
                searched[y,x] = 1
                disIm[y,x] = val

                candidates = self.neighbors((y,x),val)
                for candidate in candidates:
                    cy = candidate[1][0]
                    cx = candidate[1][1]
                    cval = candidate[0]
                    if searched[cy,cx] == 0:
                        if cval < mincost[cy][cx]:
                            mincost[cy][cx] = cval
                            heapq.heappush(frontier, candidate)

        disIm[disIm == np.inf] = -5
        disIm = disIm[1:-1,1:-1]
        disIm[pt] = 0

        return disIm


    def neighbors(self, pt, val):
        return [(val+1.414,(pt[0] + y, pt[1] + x)) if y!=0 and x!=0 else (val + 1.0,(pt[0] + y, pt[1] + x)) for y in [-1,0,1] for x in [-1,0,1]]

    def find_edges(self):
        self.edges = np.bitwise_and(self.im, canny(self.im))
        return self.edges


class JoinedSegments:
    """
    Base class which Root and Branch inherit from
    """
    pass


class Root(JoinedSegments):
    """
    The root of a joined segment. Contains a soma
    """
    def __init__(self, soma:Soma):

        self.start = soma
        self.cellid = soma.label
        self.end = soma
        self.get_length()
        self.ymin = soma.ymin
        self.ymax = soma.ymax
        self.xmin = soma.xmin
        self.xmax = soma.xmax

    def get_length(self):
        """
        see Branch.get_length
        """
        # base condition
        self.start_length = 0
        self.disIm = np.zeros(self.end.region.edges.shape)

        return 0, self.disIm

    def get_rnadistances(self, fetch=True):
        """
        See Branch.get_rnadistances
        """
        # base condition
        self.rnadists = np.array([])
        self.rnaids = np.array([])

        if fetch:
            return self.rnaids.copy(), self.rnadists.copy()

    def plot(self, final_layer=False):
        return [(self.ymin, self.xmin, self.ymax, self.xmax)], [self.end.region.im.copy()], self.end.rna_loc.copy()


class Branch(JoinedSegments):
    """
    A JoinedSegment with a process at the end.
    Builds off another JoinedSegment
    """

    def __init__(self, source:JoinedSegments, process:Process):

        self.disIm = None
        self.start_length = None
        self.source = source
        self.start = source.start
        self.cellid = source.cellid
        self.end = process
        self.get_length()

        self.ymin = min(process.ymin, source.ymin)
        self.ymax = max(process.ymax, source.ymax)
        self.xmin = min(process.xmin, source.xmin)
        self.xmax = max(process.xmax, source.xmax)

    def get_length(self):
        """
        Determine the length of the branch recursively

        ---Returns---
        self.start_length: the length of the shortest path from the soma to the
            nearest pixel in self.end
        self.disImg: disIm calculated from the nearest pixel in self.end as
            the source
        """

        # call to source branch
        sourcestartlength, sourcedisIm = self.source.get_length()

        # calculate distance between end and source
        nearestSelf, nearestSource, dist = self.end.distance_from(self.source.end)

        # calculate total length at start of end segment, and length along each pixel
        self.start_length = sourcestartlength + sourcedisIm[nearestSource] + dist
        self.disIm = self.end.region.make_disgraph(nearestSelf)

        # track points
        self.sourcept = nearestSource
        self.startpt = nearestSelf

        return self.start_length, self.disIm

    def get_rnadistances(self, fetch=True):
        """
        Determine the distance of each RNA from the soma recursively

        ---Arguments---
        fetch: whether to return values

        ---Returns---
        rnaids: the ids of all RNA along the branch
        rnadists: the distances of each RNA from the soma
        """
        # get RNA distribution up to here
        all_rna, all_rna_dists = self.source.get_rnadistances()

        # using closest pt to source calc rna distances and fetch rnaids
        selfdists = self.end.get_rnadistances(self.startpt)
        selfids = self.end.rna_index

        # modify distances to include distance along branch and add to total distribution
        self.rnadists = np.append(all_rna_dists, selfdists + self.start_length)
        self.rnaids = np.append(all_rna, selfids)

        # only make copy if asked
        if fetch:
            return self.rnaids.copy(), self.rnadists.copy()

    def plot(self, final_layer=True):
        bboxes, images, rna_locs = self.source.plot(final_layer=False)
        bboxes.append([self.end.ymin, self.end.xmin, self.end.ymax, self.end.xmax])
        images.append(self.end.region.im)
        rna_locs = np.vstack((rna_locs, self.end.rna_loc.copy()))

        if not final_layer:
            return bboxes, images, rna_locs

        else:
            image = np.zeros((self.ymax - self.ymin, self.xmax - self.xmin))
            for i, im in enumerate(images):
                bbox = bboxes[i]
                image[(bbox[0]-self.ymin):(bbox[2]-self.ymin), (bbox[1]-self.xmin):(bbox[3]-self.xmin)] += im*(i + 1)


        plotting.implot(image,cmap='cubehelix',vmax=len(bboxes)+2,show=False)
        rna_locs[:,0] -= self.ymin
        rna_locs[:,1] -= self.xmin

        plt.scatter(rna_locs[:,1], rna_locs[:,0], c='#DC96FF', s=0.25)


def buildProcesses(image, rna):
    """
    Builds Processes from the specified image and rna dataset

    ---Arguments---
    image: labeled image containing the processes
    rna: dataset containing the locations and barcodes of rna in the image

    ---Returns---
    List of all processes in the image as Processes
    """
    props = regionprops(image)

    rnaxs = np.array(rna['global_x'])
    rnays = np.array(rna['global_y'])
    rnalocs = np.array([rnays,rnaxs]).T
    rnabc = np.array(rna['barcode_id'])

    rnaxpx = np.array(np.floor(rnaxs),dtype=np.int)
    rnaypx = np.array(np.floor(rnays),dtype=np.int)

    rna_index = image[rnaypx,rnaxpx]

    processes = {}

    n = 5

    for prop in tqdm(props, desc="building processes"):
        process = Process(prop)
        process.addRNA(rna_index==prop.label, rnalocs, rnabc)

        processes[process.label]=process

    return processes


def buildSomas(image, rna):
    """
    Builds Somas from the specified image and rna dataset

    ---Arguments---
    image: labeled image containing the somas
    rna: dataset containing the locations and barcodes of rna in the image

    ---Returns---
    List of all somas in the image as Somas
    """
    props = regionprops(image)

    rnaxs = np.array(rna['global_x'])
    rnays = np.array(rna['global_y'])
    rnalocs = np.array([rnays,rnaxs]).T
    rnabc = np.array(rna['barcode_id'])

    rnaxpx = np.array(np.floor(rnaxs),dtype=np.int)
    rnaypx = np.array(np.floor(rnays),dtype=np.int)

    rna_index = image[rnaypx,rnaxpx]

    somas = {}

    for prop in tqdm(props):
        soma = Soma(prop)
        soma.addRNA(rna_index==prop.label, rnalocs, rnabc)

        somas[soma.label] = soma

    return somas
