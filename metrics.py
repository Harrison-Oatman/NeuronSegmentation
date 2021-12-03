from plotting.py import embedscatter
import numpy as np

def align_RNA(RNA_a, RNA_b):
    """
    Aligns two sets of annotated RNA

    ---Arguments---
    RNA_a, RNA_b: sets of RNA which have been annotated

    ---Returns---
    ab_a: aligned RNA with A's annotations
    ab_b: aligned RNA with B's annotations
    a_to_b: conversion from A cell names to B cell names
    b_to_a: conversion from B cell names to A cell names
    """
    # returns RNA datasets with shared RNA aligned
    # returns list to convert between cell index in a and cell index in b
    xloc_a = np.array(RNA_a["global_x"])
    xloc_b = np.array(RNA_b["global_x"])

    # find the intersection of the arrays, so only RNA inclusive to both arrays is here
    intersect, a_ids, b_ids = np.intersect1d(xloc_a,xloc_b, return_indices=True)

    # take only the intersection
    ab_a = pd.DataFrame(RNA_a.iloc[a_ids,:])
    ab_b = pd.DataFrame(RNA_b.iloc[b_ids,:])

    # get new cell indices
    cell_a = np.array(ab_a["cell_index"])
    cell_b = np.array(ab_b["cell_index"])

    # find the maximum cell which was labeled
    cell_max_a = np.max(cell_a) + 1
    cell_max_b = np.max(cell_b) + 1

    # create matrix to determine corresponding labelings
    decode_matrix = np.zeros((int(cell_max_a), int(cell_max_b)))
    for i in range(len(ab_a)):
        j = cell_a[i]
        k = cell_b[i]
        decode_matrix[j,k] += 1

    a_to_b = list(np.argmax(decode_matrix,1))
    b_to_a = list(np.argmax(decode_matrix,0))

    return ab_a, ab_b, a_to_b, b_to_a

"""
categorization functions
"""

def process(cell, process): # RNA is contained in a labeled process
    return process > 0

def cell(cell, process): # RNA is contained in a labeled cell
    return cell > 0

def soma(cell,process): # RNA is contained in a labeled soma
    return np.bitwise_and(cell > 0, process == 0)

def proximal(cell, process): # RNA is contained in a process directly next to a soma
    return np.bitwise_and(cell > 0, np.bitwise_and(process < 10, process > 0))

def distal(cell, process): # RNA is contained in a process at least two segments from a soma
    return np.bitwise_and(cell > 0, process > 99)

def one(cell, process): # RNA exists
    return True

def cell_accuracy(RNA_a, RNA_b, first=process, second=soma,n=10**5):
    """
    ---Arguments---
    RNA_a: RNA dataset with ground truth labels
    RNA_b: RNA dataset with labels to test
    b_to_a: (not currently in use) converts between cell labels in b and a
    first: bool function to determine how to select the first RNA in a pair
    second: bool function to determine how to select the second RNA in a pair
    n: number of trials -- how many pairs should we sample?


    first and second are two functions which take in a cell index and process index and return a Bool--
    for example, the function "process" returns True if the RNA has a nonzero process index
    """

    """
    I start by getting the cell and process indices from the dataset, because I
    prefer to work with numpy arrays
    """
    # work with np arrays, get cell indices
    cell_id_a = np.array(RNA_a["cell_index"])
    cell_id_b = np.array(RNA_b["cell_index"])

    # convert cell indices from b to a, not used
    # converted_b = np.array([b_to_a[i] for i in cell_id_b])

    # get process indices
    process_id_a = np.array(RNA_a["process_index"])
    process_id_b = np.array(RNA_b["process_index"])

    """
    Here we choose candidate RNA to be considered for the pair
    Using the ground truth, we see which RNA are marked True by the first function,
    and which RNA are marked True by the second function
    """
    # get set of RNA which fit into first and second class - ground truth
    first_truth_a = first(cell_id_a, process_id_a)
    second_truth_a = second(cell_id_a, process_id_a)

    # get the cells corresponding to those RNA
    first_cells_a = cell_id_a[first_truth_a]
    second_cells_a = cell_id_a[second_truth_a]

    """
    We want to make sure that we're taking RNA from the same cell, so to speed up this
    process we sort the indices which fall into the second category by cell
    """
    # make a list of lists, where each sublist contains the indices of second type RNA
    second_pools = [np.argwhere(second_cells_a == i) for i in range(np.max(cell_id_a)+1)]
    second_pools = [np.array([b[0] for b in a]) if len(a) > 0 else a for a in second_pools]

    """
    Here we pick our random pairs, by choosing a random first category RNA, and then choosing a random
    second category RNA from the same cell
    """
    # pick a random id from first list, and pick random id from corresponding cell in second list
    first_ids = np.random.choice(np.arange(len(first_cells_a)),size=n)
    second_ids = [np.random.choice(second_pools[i]) for i in first_cells_a[first_ids]]

    """
    Now we determine the process index of each of those RNA according to a (the ground truth)
    """
    # get index of RNAs
    indices = np.arange(len(first_truth_a))
    first_indices = indices[first_truth_a]
    first_indices = first_indices[first_ids]
    second_indices = indices[second_truth_a]
    second_indices = second_indices[second_ids]


    # get processes of a RNA
    first_process_a = process_id_a[first_indices]
    second_process_a = process_id_a[second_indices]


    """
    Now we determine the process index and cell index of those RNA according to b (the one being tested)
    """
    # get cells, processes of RNA according to b
    first_cells_b = cell_id_b[first_indices]
    first_process_b = process_id_b[first_indices]
    second_cells_b = cell_id_b[second_indices]
    second_process_b = process_id_b[second_indices]

    """
    Here we make sure that person a labeled both RNA in each pair
    This should be true because the functions filtered for RNA which were labeled
    """
    first_is_cell_a = cell(first_cells_a[first_ids], first_process_a)
    second_is_cell_a = cell(second_cells_a[second_ids], second_process_a)
    # print(f"sanity check (should be 1.0): {np.average(np.bitwise_and(first_is_cell_a, second_is_cell_a))}")

    """
    Here we figure out how many pairs were labeled by person b
    Iff both of the RNA have been labeled as soma or process then True
    """
    first_is_cell_b = cell(first_cells_b, first_process_b)
    second_is_cell_b = cell(second_cells_b, second_process_b)
    classified = np.bitwise_and(first_is_cell_b, second_is_cell_b)

    # print(f"probability both RNA labeled & {np.average(classified)} \\\\")


    """
    Here we figure out if person b gave the same labels as person b
    """
    first_truth_b = first(first_cells_b, first_process_b)
    second_truth_b = second(second_cells_b, second_process_b)
    correctly_classified = np.bitwise_and(first_truth_b, second_truth_b)
    # print(f"probability both RNA correctly classified & {np.average(correctly_classified)} \\\\")
    # print(f"probability both RNA correctly classified|labeled & {np.average(correctly_classified[classified])} \\\\" )

    """
    Here we figure out if person b assigned the pair of RNA to the same cell
    """
    same_cell = second_cells_b == first_cells_b
    same_cell = np.bitwise_and(same_cell, classified)

    # print(f"probability both RNA assigned to same cell & {np.average(same_cell)} \\\\")
    # print(f"probability both RNA assigned to same cell|labeled & {np.average(same_cell[classified])} \\\\")

    """
    This determines if person b assigned the pair to the same cell with the correct labels
    """
    score = np.bitwise_and(same_cell, correctly_classified)
    # print(f"probability both RNA correctly classified and assigned to same cells & {np.average(score)} \\\\")
    # print(f"probability both RNA correctly classified and assigned to same cell | labeled & {np.average(score[classified])} \\\\")


    scores = {}
    scores["avg both given label"] = np.average(classified)
    scores["avg both correctly classfied"] = np.average(correctly_classified)
    scores["avg both correctly classified if labeled"] = np.average(correctly_classified[classified])
    scores["avg same cell"] = np.average(same_cell)
    scores["avg same cell if labeled"] = np.average(same_cell[classified])
    scores["avg both correctly classified and assigned"] = np.average(score)
    scores["avg both correclty classified and assigned if labeled"] = np.average(score[classified])
    # shorthand
    scores["lab"] = classified
    scores["cc"] = correctly_classified
    scores["ccgl"] = correctly_classified[classified]
    scores["ss"] = same_cell
    scores["ssgl"] = same_cell[classified]
    scores["ccss"] = score
    scores["ccssgl"] = score[classified]

    return scores, first_indices, second_indices

def classification_accuracy(RNA_a, RNA_b, metric, base_condition = one):
    """
    Looks at all RNA in A and B which agree on base_condition, calculates the
    rate at which they agree on metric.
    """
    cell_id_a = np.array(RNA_a["cell_index"])
    cell_id_b = np.array(RNA_b["cell_index"])

    process_id_a = np.array(RNA_a["process_index"])
    process_id_b = np.array(RNA_b["process_index"])

    base_a = base_condition(cell_id_a, process_id_a)
    base_b = base_condition(cell_id_b, process_id_b)

    agreed = np.bitwise_and(base_a,base_b)

    cell_base_a = cell_id_a[agreed]
    cell_base_b = cell_id_b[agreed]

    process_base_a = process_id_a[agreed]
    process_base_b = process_id_b[agreed]

    metric_a = metric(cell_base_a, process_base_a)
    metric_b = metric(cell_base_b, process_base_b)

    a_specificity = np.average(metric_b[metric_a == False] == False)
    a_sensitivity = np.average(metric_b[metric_a])
    b_specificity = np.average(metric_a[metric_b == False] == False)
    b_sensitivity = np.average(metric_a[metric_b])
    # print(f"specifificty: {specificity}")
    # print(f"sensitivity: {sensitivity}")
    # print(f"{specificity:0.5}")
    # print(f"{sensitivity:0.5}")
    return [b_sensitivity, b_specificity, a_sensitivity, b_specificity], agreed, metric_a, metric_b

def plot_pair_errors(rna, scores, pair="process-soma", test="ss", bbs=None, im=None):
    """
    Takes in score results and plots errors in pair classifications
    """
    first_indices = scores[pair][1]
    second_indices = scores[pair][2]
    vals = scores[pair][0][test]

    xs = np.array(rna["global_x"])
    ys = np.array(rna["global_y"])

    firstxs = xs[first_indices]
    firstys = ys[first_indices]

    secondxs = xs[second_indices]
    secondys = ys[second_indices]

    allxs = np.append(firstxs, secondxs)
    allys = np.append(firstys, secondys)
    allscores = np.append(vals, vals)

    embedscatter(im, firstxs, firstys, c=vals, bbs=bbs)

def plot_classification_accuracy(rna, scores, metric="process", im=None, bbs=None):
    """
    takes in score results and plots errors in categorical classifications
    """
    agreed = scores[metric][1]

    xs = np.array(rna["global_x"])[agreed]
    ys = np.array(rna["global_y"])[agreed]

    metric_a = scores[metric][2]
    metric_b = scores[metric][3]

    col = np.array(["k", "r", "orange", "g"])

    scores = 2*metric_a + metric_b
    c = col[scores]



    include = scores > 0

    decimate = np.array(np.random.choice(len(c[include]), size=int(len(c[include])/10)))

    embedscattercolor(im, xs[include][decimate], ys[include][decimate], c[include][decimate], bbs=bbs)

def accuracy_set(RNA_a,RNA_b):
    """
    Takes several metrics
    """
    scores_dict = {}
    scores_dict["cell-cell"] = cell_accuracy(RNA_a,RNA_b,cell,cell)
    scores_dict["soma-soma"] = cell_accuracy(RNA_a,RNA_b,soma,soma)
    scores_dict["process-process"] = cell_accuracy(RNA_a,RNA_b,process,process)
    scores_dict["process-soma"] = cell_accuracy(RNA_a,RNA_b,process,soma)
    scores_dict["proximal-soma"] = cell_accuracy(RNA_a,RNA_b,proximal,soma)
    scores_dict["distal-soma"] = cell_accuracy(RNA_a,RNA_b,distal,soma)
    scores_dict["soma"] = classification_accuracy(RNA_a,RNA_b, soma)
    scores_dict["process"] = classification_accuracy(RNA_a,RNA_b, process)
    scores_dict["proximal"] = classification_accuracy(RNA_a,RNA_b, proximal)
    scores_dict["distal"] = classification_accuracy(RNA_a,RNA_b, distal)

    return scores_dict
