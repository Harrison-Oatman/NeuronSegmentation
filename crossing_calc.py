import numpy as np
from tqdm import tqdm


def get_point_neighbors(involved, label_image):

    pt_neighbors = {}

    for i, instance_list in tqdm(enumerate(involved), desc='evaluating neighbor relationships'):
        processes = np.unique(label_image[instance_list[:,0], instance_list[:,1]])
        for process in processes:
            if pt_neighbors.get(process) is None:
                pt_neighbors[process] = {}
            for process_two in processes:
                pt_neighbors[process][process_two] = True

    return pt_neighbors


def do_crossing_calc(cp_involved, bp_involved, label_image):
    branching_neighbors = get_point_neighbors(bp_involved, label_image)
    crossing_neighbors = get_point_neighbors(cp_involved, label_image)
    return crossing_neighbors, branching_neighbors


def get_neighbor_mat(neighbors, n):
    processes = list(neighbors.keys())
    neiArray = np.zeros((n, n))

    for i in processes:
        for j in neighbors[i].keys():
            neiArray[i][j] = neighbors[i][j]

    return neiArray


def get_crossing_mats(cp_involved, bp_involved, label_image):
    crossing_neighbors, branching_neighbors = do_crossing_calc(cp_involved, bp_involved, label_image)
    n = np.max(label_image + 1)
    branch_mat = get_neighbor_mat(branching_neighbors, n)
    cross_mat = get_neighbor_mat(crossing_neighbors, n)
    return cross_mat, branch_mat
