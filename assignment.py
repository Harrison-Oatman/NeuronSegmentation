import numpy as np
from tqdm import tqdm
import distance_calc
import roots_calc
import crossing_calc

def get_rna_process(rna, process_image):
    def get_process_index(x, y):
        return process_image[int(np.floor(y)), int(np.floor(x))]

    xs = np.array(rna['global_x'])
    ys = np.array(rna['global_y'])

    v_process = np.vectorize(get_process_index, otypes=[np.int])

    process_index = v_process(xs, ys)

    return process_index


def clean_cost_array(arr):
    costArray = arr.copy()
    costArray[0, :] = np.inf
    costArray[:, 0] = np.inf
    return costArray

def clean_nei_array(arr):
    costArray = arr.copy()
    costArray[0, :] = 0
    costArray[:, 0] = 0
    return costArray


def linear_lvl(lvl):
    return lvl * 3


def hierarchy_join(assigned, unassigned, cost_mat, assignments=None, levelfn=linear_lvl):
    if assignments == None:
        # self assignment indicates process_test is a root
        assignments = [i for i in range(cost_mat.shape[0])]
        levels = np.array([0 for i in range(cost_mat.shape[0])])

    t = tqdm(total=len(unassigned), desc="joining segments")

    while len(unassigned) > 0:

        t.update(1)

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

    t.close()

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


def secret_sauce(dis_arr, cross_arr, branch_arr):
    cleaned_dis_arr = clean_cost_array(dis_arr)
    cleaned_cross_arr = clean_nei_array(cross_arr)
    cross_times_dis = cleaned_dis_arr.copy()
    cross_times_dis[cleaned_cross_arr == 0] = 0
    cleaned_branch_arr = clean_nei_array(branch_arr)

    cost_one = cleaned_dis_arr - 1.5 * (cross_times_dis)

    return cost_one


def start_to_end(process_image, soma_image, cp_involved, bp_involved, rna):
    distance_dict = distance_calc.image_to_distances(process_image)

    n = np.max(process_image) + 1
    dis_array = distance_calc.get_distance_mat(distance_dict,n)
    cross_arr, branch_arr = crossing_calc.get_crossing_mats(cp_involved, bp_involved, process_image)
    print(np.sum(cross_arr, axis=1))

    cleaned_cost_arr = secret_sauce(dis_array, cross_arr, branch_arr)

    roots, parent_cells = roots_calc.get_roots_and_parents(process_image, soma_image)
    assigned, unassigned = roots_calc.assignments_from_roots(roots)

    assignments = hierarchy_join(assigned, unassigned, cleaned_cost_arr)

    process_labels = get_process_label(assignments, parent_cells)
    cell_image = get_cell_image(process_image, assignments, parent_cells, soma_image)
    new_rna = get_new_rna(rna, process_image, process_labels, cell_image)
    return assignments, process_labels, cell_image, new_rna
