import numpy as np
from skimage.morphology import disk, dilation

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
