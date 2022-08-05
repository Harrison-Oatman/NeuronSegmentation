import numpy as np
import tensorflow as tf
from ..const import *
from .trainingsamples import load_branch_json_samples, write_branch_json_samples
from ..utils.transforms import BranchTransforms
import math


class BranchSplitter:

    def __init__(self, train_dir=BRANCH_SAMPLE_PATH, split=TRAIN_VAL_SPLIT,
                 overwrite_split_labels=False):

        json_path = train_dir + "samples.json"

        self.examples = load_branch_json_samples(json_path)
        self.examples = {int(k): v for k, v in self.examples.items()}
        self.n_examples = len(self.examples.keys())

        if overwrite_split_labels:
            self.split_ids = {x: [] for x in ["train", "val"]}
            unlabeled_ids = [i for i in self.examples]
        else:
            self.split_ids = {split: [i for i in self.examples if
                                      self.examples[i].get("split", None) == split]
                              for split in ["train", "val"]}
            unlabeled_ids = [i for i in self.examples if self.examples[i].get("split", None) is None]
        n_unlabeled = len(unlabeled_ids)

        to_train = math.floor(self.n_examples * split) - len(self.split_ids["train"])

        if to_train < 0 or to_train > n_unlabeled:
            print("not enough unlabeled samples to successfully split, try setting"
                  "'overwrite_split_labels' to True")
            to_train = int(np.clip(to_train, 0, n_unlabeled))

        np.random.shuffle(unlabeled_ids)
        new_trains = unlabeled_ids[:to_train]
        new_vals = unlabeled_ids[to_train:]

        self.split_ids["train"].extend(new_trains)
        self.split_ids["val"].extend(new_vals)

        for i in new_trains:
            self.examples[i]["split"] = "train"

        for i in new_vals:
            self.examples[i]["split"] = "val"

        write_branch_json_samples(self.examples, train_dir + "samples.json")

    def get_samples(self, split):
        return [self.examples[i] for i in self.split_ids[split]]


class BranchTrainLoader:

    def __init__(self, split, splitter: BranchSplitter,
                 batch_size=BRANCH_BATCH_SIZE_TRAIN, shuffle=True):

        self.splitter = splitter
        self.samples = self.splitter.get_samples(split)
        self.batch_size = batch_size
        self.i = 0
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
        self.i = 0
        return self

    def __next__(self):
        if self.i + self.batch_size > len(self.samples):
            raise StopIteration

        batch = BranchBatch(self.samples[self.i:self.i + self.batch_size])
        self.i += self.batch_size

        return batch


class BranchBatch:

    def __init__(self, samples):
        self.samples = samples
        self.n = len(self.samples)
        self.poms = np.array([np.load(sample["poms"]) for sample in self.samples])
        self.inputs = np.array([np.load(sample["input"]) for sample in self.samples])
        self.labels = np.array([sample["label"] for sample in self.samples])
        self.rna_vecs = np.array([np.load(sample["rna_vecs"]) for sample in self.samples])
        self.pairs = np.array([sample["pair"] for sample in self.samples])

        self.transformer = BranchTransforms()

    def data(self):
        return self.transformer.preprocess(self.inputs, self.poms, self.labels)

    def sample_weights(self):
        avg_true = np.average(self.labels)
        false_weight = avg_true / (1 - avg_true + 0.000001)
        sample_weights = [false_weight if a < 0.5 else 1 for a in self.labels]
        return sample_weights

