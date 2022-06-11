import math
import numpy as np
from .. import const
from ..utils import sampling


class Splitter:

    def __init__(self, train_dir=const.TRAINING_DIR, split=const.TRAIN_VAL_SPLIT,
                 overwrite_split_labels=False):
        """
        args:
            train_dir: location of examples.json file
            split: fraction of training examples
            overwrite_split: whether to keep existing labels
        """

        self.examples = sampling.load_json_examples(train_dir+"examples.json")
        self.n_examples = len(self.examples.keys())
        if overwrite_split_labels:
            self.split_ids = {x: [] for x in ["train", "val"]}
            unlabeled_ids = [i for i in self.examples]
        else:
            self.split_ids = {split: [i for i in self.examples if
                                      self.examples[i].split == split]
                              for split in ["train", "val"]}
            unlabeled_ids = [i for i in self.examples if self.examples[i].split is None]
        n_unlabeled = len(unlabeled_ids)

        to_train = math.floor(self.n_examples*split) - len(self.split_ids["train"])

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
            self.examples[i].split = "train"

        for i in new_vals:
            self.examples[i].split = "val"

        sampling.write_json_examples(train_dir + "examples.json", self.examples)

    def get_samples(self, split):
        return [self.examples[i] for i in self.split_ids[split]]
