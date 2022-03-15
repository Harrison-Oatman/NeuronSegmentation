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


def get_rna_process(rna, process_image):
    def get_process_index(x, y):
        return process_image[int(np.floor(y)), int(np.floor(x))]

    xs = np.array(rna['global_x'])
    ys = np.array(rna['global_y'])

    v_process = np.vectorize(get_process_index, otypes=[np.int])

    process_index = v_process(xs, ys)

    return process_index


# def get_process_cluster(process_image, rna):
#     RNAx = rna["global_x"]
#     RNAy = rna["global_y"]
#     RNAid = rna['barcode_id']
#     RNAcol = [barcodeColor[id] for id in RNAid]
#     processIndex = get_rna_process(rna, process_image)
#
#     clusters = np.unique(RNAcol)
#     processCluster = {}
#
#     # print(processIndex)
#     for i, process_test in enumerate(processIndex):
#         # print(process_test)
#         if process_test not in processCluster:
#             processCluster[process_test] = {}
#             for cluster in clusters:
#                 processCluster[process_test][cluster] = 0
#         processCluster[process_test][RNAcol[i]] += 1
#
#     return processCluster


# def getScaledClusterVal(process_image, clusters, color):
#     n = np.max(list(clusters.keys()))
#     relval = np.zeros(n+10)
#     for process_test in range(n):
#         if process_test in clusters:
#             tot = np.sum(list(clusters[process_test].values()))
#             # print(tot)
#             col = np.sum(clusters[process_test][color])
#             relval[process_test] = col/tot
#
#     relval = relval + 0.05
#
#     relval[0] = 0
#
#     def colorofprocess(process_test):
#         return relval[process_test]
#
#     print(np.max(process_image))
#     print(n)
#
#     v_get_val = np.vectorize(colorofprocess)
#
#     return v_get_val(process_image)


# def plotClusterImage(processIamge, clusters, color, cmap, bbs):
#     colored = getScaledClusterVal(process_image, clusters, color)
#     embedimg(colored, bbs, cmap=cmap,name="process_cluster_proportion_" + color)


# def getScaledClusters(clusters):
#     scaledClusters = {}
#     for i in clusters:
#         scaledClusters[i] = {}
#         tot = 5
#         for c in clusters[i]:
#             tot += clusters[i][c]
#         for c in clusters[i]:
#             scaledClusters[i][c] = clusters[i][c]/tot
#     return scaledClusters


# def getClusterDis(clusters, neighbors):
#     sc = getScaledClusters(clusters)
#
#     def distance(i, j):
#         if i in sc and j in sc:
#             ic = sc[i]
#             jc = sc[j]
#             diff = 0
#             for c in ic:
#                 if c is "black":
#                     continue
#                 diff += abs(ic[c] - jc[c])
#             return diff
#         else:
#             return 0.5
#
#     distances = {}
#
#     for i in neighbors.keys():
#         distances[i] = {}
#         for j in neighbors[i]:
#             dist = distance(i, j)
#             distances[i][j] = dist
#
#     return distances


def get_distance_mat(distances):
    processes = list(distances.keys())
    n = np.max(processes)
    disArray = np.zeros((n + 1, n + 1))
    disArray += np.inf
    for i in processes:
        for j in distances[i].keys():
            disArray[i][j] = distances[i][j]

    return disArray


def overlap_sets(processes, soma):
    roots = np.array([True for _ in range(np.max(processes) + 1)])
    parentCells = np.zeros(np.max(processes)+1)

    ys, xs =  np.where(np.logical_and(processes != 0, soma != 0))

    sets = [set() for _ in range(np.max(processes) + 1)]

    for y, x in zip(ys, xs):
        sets[processes[y,x]].add(soma[y,x])

    for process, overlap in enumerate(sets,0):
        if len(overlap) == 0:
            roots[process] = False
        else:
            bestmatch = 0
            most = 0
            for somaid in list(overlap):
                val = np.sum(np.logical_and(processes==process,soma==somaid))
                if val > most:
                    bestmatch = somaid
                    most = val
            parentCells[process] = bestmatch

    return roots, parentCells


def get_roots_and_parents(process_image, soma_image, rad=3):
    dilated_soma = dilation(soma_image, disk(rad))
    return overlap_sets(process_image, dilated_soma)


def assignments_from_roots(roots):
    rang = np.arange(len(roots))
    return np.array(rang[roots]), np.array(rang[roots == False])


def clean_cost_array(arr):
    costArray = arr.copy()
    costArray[0, :] = np.inf
    costArray[:, 0] = np.inf
    return costArray


def linear_lvl(lvl):
    return lvl * 3


def hierarchy_join(assigned, unassigned, cost_mat, assignments=None, levelfn=linear_lvl):
    if assignments == None:
        # self assignment indicates process_test is a root
        assignments = [i for i in range(cost_mat.shape[0])]
        levels = np.array([0 for i in range(cost_mat.shape[0])])

    while len(unassigned) > 0:
        # take submatrix of possible pairings
        subMat = cost_mat[assigned, :]
        subMat = subMat[:, unassigned]

        # find the pairing with minimum cost
        pid, cid = np.unravel_index(np.argmin(subMat), subMat.shape)

        # assignment[child] of child is parent
        assignments[unassigned[cid]] = assigned[pid]

        # level assignment is one greater than level of parent
        levels[unassigned[cid]] = levels[assigned[pid]] + 1

        # correct cost_mat to account for level
        cost_mat[unassigned[cid], :] += levelfn(levels[unassigned[cid]])

        # assign child and then rm from unassigned
        assigned = np.append(assigned, unassigned[cid])
        unassigned = np.delete(unassigned, cid)

        # sort assigned array
        assigned = np.sort(assigned)

    return assignments


def get_process_label(assignments, parent_cells):
    n = len(assignments)
    m = np.max(parent_cells) + 1
    processIndex = np.ones(n, dtype=np.int)
    cellCount = np.ones(n, dtype=np.int)
    processLabel = np.zeros(n, dtype=np.int)
    processAssigned = np.zeros(n, dtype=np.bool)
    while np.average(processAssigned) < 1.0:
        for i in range(n):
            if not processAssigned[i]:
                if assignments[i] == i:
                    processLabel[i] = cellCount[int(parent_cells[i])]
                    cellCount[int(parent_cells[i])] += 1
                    processAssigned[i] = True
                elif processAssigned[assignments[i]]:
                    label_body = processLabel[assignments[i]]
                    label_tail = processIndex[assignments[i]]
                    processAssigned[i] = True
                    processLabel[i] = label_body * 10 + label_tail
                    processIndex[assignments[i]] += 1

    processLabel[0] = 0
    return processLabel


def cell_from_assignment(cell_processes, assignments, maxiter=10):
    poots = assignments
    for _ in range(maxiter):
        poots = np.array([poots[id] for id in poots])

    print(poots)

    parentcells = np.array([cell_processes[i] for i in poots])

    return parentcells


def get_cell_image(process_image, assignments, process_cells, soma_image):
    cells = cell_from_assignment(process_cells, assignments, 30)
    cells[0] = 0

    def get_cell(p):
        return cells[p]

    v_get_cell = np.vectorize(get_cell)

    return v_get_cell(process_image) + soma_image


def get_new_rna(rna, process_image, process_labels, cell_image):
    def getCell(x, y):
        return cell_image[int(np.floor(y)), int(np.floor(x))]

    def getProcessIndex(x, y):
        return process_labels[process_image[int(np.floor(y)), int(np.floor(x))]]

    newRNA = rna.copy()
    xs = np.array(newRNA['global_x'])
    ys = np.array(newRNA['global_y'])

    v_cell = np.vectorize(getCell, otypes=[np.int])
    v_process = np.vectorize(getProcessIndex, otypes=[np.int])

    cell_index = v_cell(xs, ys)
    process_index = v_process(xs, ys)

    newRNA['process_index'] = process_index
    newRNA['cell_index'] = cell_index

    return newRNA


def start_to_end(process_image, soma_image, rna):
    props = regionprops(process_image)
    neighbors = get_neighbors(props, process_image)
    wheres, edges = get_wheres(process_image)
    distances, closests = get_distance(neighbors, wheres, edges)
    # processCluster = get_process_cluster(process_image, rna)
    # clusterDis = getClusterDis(processCluster, neighbors)
    dis_array = get_distance_mat(distances)
    # clusterDisArray = get_distance_mat(clusterDis)
    roots, parent_cells = get_roots_and_parents(process_image, soma_image)
    assigned, unassigned = assignments_from_roots(roots)
    cleaned_cost_arr = clean_cost_array(dis_array)
    assignments = hierarchy_join(assigned, unassigned, cleaned_cost_arr)
    process_labels = get_process_label(assignments, parent_cells)
    cell_image = get_cell_image(process_image, assignments, parent_cells, soma_image)
    new_rna = get_new_rna(rna, process_image, process_labels, cell_image)
    return assignments, process_labels, cell_image, new_rna
