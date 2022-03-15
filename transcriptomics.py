import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF


def get_transcriptomes(rna):

    max_cell = rna["cell_index"].max()
    max_rna = rna['barcode_id'].max()

    cell_transcriptomes = {x: np.zeros(max_rna + 1) for x in range(max_cell + 1)}

    def send_point(pt):
        # print(pt["cell_index"])
        cell_transcriptomes[int(pt["cell_index"])][int(pt["barcode_id"])] += 1

    rna.apply(send_point, axis=1)

    pops = []
    for cell_id in cell_transcriptomes.keys():
        if np.sum(cell_transcriptomes[cell_id]) < 1:
            pops.append(cell_id)

    for cell_id in pops:
        del cell_transcriptomes[cell_id]

    for cell_id in cell_transcriptomes.keys():
        cell_transcriptomes[cell_id] /= np.sum(cell_transcriptomes[cell_id])

    del cell_transcriptomes[0]

    return cell_transcriptomes


def apply_nmf(transcriptomes:dict, n_components=None):
    X = [item[1] for item in transcriptomes.items()]
    X = np.vstack(X)
    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    return nmf
