from skimage.measure import regionprops
import numpy as np
from scipy.spatial.distance import cdist
from plotting import embedimg
from skimage.morphology import disk, dilation

def get_distance(neighbors, wheres, edges):
    def distance(i, j):
        dist = cdist(edges[i], edges[j])
        k, m = np.unravel_index(np.argmin(dist), dist.shape)
        # dist = np.argmin(dist)
        # return dist.min()
        # print(dist.shape)
        return edges[i][k], edges[j][m], dist.min()

    distances = {}
    closests = {}

    for i in neighbors.keys():
        distances[i] = {}
        closests[i] = {}
        for j in neighbors[i]:
            iloc, jloc, dist = distance(i, j)
            distances[i][j] = dist
            closests[i][j] = iloc

    return distances, closests


def get_neighbors(props, process_image, max_dist=20):
    n, m = process_image.shape
    neighbors = {}
    for prop in props:
        d = max_dist
        i = prop.label
        while True:
            bbox = prop.bbox

            ymin = max(0, bbox[0] - d)
            ymax = min(n - 1, bbox[2] + d - 1)
            xmin = max(0, bbox[1] - d)
            xmax = min(m - 1, bbox[3] + d - 1)

            nbs = np.unique(process_image[ymin:ymax, xmin:xmax])

            nbs = nbs[nbs != 0]
            nbs = nbs[nbs != i]

            if len(nbs > 0):
                neighbors[i] = nbs
                break
            else:
                d += 10
    return neighbors


def get_wheres(pimage):
    processes = np.unique(pimage)
    processes = processes[processes != 0]
    n, m = pimage.shape
    wheres = {}
    edges = {}
    for i in processes:
        wheres[i] = []
        edges[i] = []

    for a in np.argwhere(pimage):
        wheres[pimage[a[0], a[1]]].append([a[0], a[1]])

    for i in processes:
        for a in wheres[i]:
            ymin = max(0, a[0] - 1)
            ymax = min(n - 1, a[0] + 1)
            xmin = max(0, a[1] - 1)
            xmax = min(m - 1, a[1] + 1)
            if pimage[ymin, a[1]] != i:
                edges[i].append(a)
            elif pimage[ymax, a[1]] != i:
                edges[i].append(a)
            elif pimage[a[0], xmin] != i:
                edges[i].append(a)
            elif pimage[a[0], xmax] != i:
                edges[i].append(a)

        wheres[i] = np.array(wheres[i])
        edges[i] = np.array(edges[i])

    return wheres, edges


def get_distance_mat(distances, n):
    processes = list(distances.keys())
    disArray = np.zeros((n, n))
    disArray += np.inf
    for i in processes:
        for j in distances[i].keys():
            disArray[i][j] = distances[i][j]

    return disArray


def image_to_distances(process_image):
    props = regionprops(process_image)
    neighbors = get_neighbors(props, process_image)
    wheres, edges = get_wheres(process_image)
    distances, closests = get_distance(neighbors, wheres, edges)
    return distances
