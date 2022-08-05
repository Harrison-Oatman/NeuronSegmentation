from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..const import *
from .branchloader import BranchBatch, BranchTrainLoader


class PCARunner:

    def __init__(self, rna_folder=PCA_SAVE_PATH):
        self.rna_vecs = np.load(rna_folder + "rows.npy")
        self.pca = PCA(n_components=40)

        self.get_pca()

    def get_pca(self):
        self.pca.fit(self.rna_vecs)

    def batch_apply(self, batch: BranchBatch):
        return [self.pca.transform(batch.rna_vecs[i]) for i in range(batch.n)]


class PCAController:

    def __init__(self, branchloader: BranchTrainLoader):
        self.loader = branchloader
        self.pcarunner = PCARunner()
        self.hitscores = np.zeros((0, N_COMPONENTS))
        self.misscores = np.zeros((0, N_COMPONENTS))

    def run(self):
        for batch in tqdm(self.loader):
            batch_trans = self.pcarunner.batch_apply(batch)
            for i, scores in enumerate(batch_trans):
                transone = scores[batch.pairs[i][1][0]]
                transtwo = scores[batch.pairs[i][1][1]]
                diff = np.abs(transone - transtwo)

                if batch.labels[i] > 0.5:
                    self.hitscores = np.vstack((self.hitscores, diff))

                else:
                    self.misscores = np.vstack((self.misscores, diff))

    def display_scores(self):
        ev = self.pcarunner.pca.explained_variance_ratio_
        fig = plt.figure(figsize=(10, 10))
        hx = np.mgrid[0:len(self.hitscores), 0:40]
        sns.scatterplot(x=hx, y=self.hitscores)
        mx = np.mgrid[0:len(self.misscores), 0:40]
        sns.scatterplot(x=mx, y=self.misscores)
        havg = np.average(self.hitscores, axis=-1)
        mavg = np.average(self.misscores, axis=-1)
        sns.plot(x=np.arange(40), y=ev)
        sns.plot(x=np.arange(40), y=havg, label="hit avg")
        sns.plot(x=np.arange(40), y=mavg, label="miss avg")
        plt.show()



