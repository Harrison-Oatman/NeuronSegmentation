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
            self.train_ids = []
            self.val_ids = []
            unlabeled_ids = [i for i in self.examples]
        else:
            self.train_ids = [i for i in self.examples if self.examples[i].split == "train"]
            self.val_ids = [i for i in self.examples if self.examples[i].split == "val"]
            unlabeled_ids = [i for i in self.examples if self.examples[i].split is None]
        n_unlabeled = len(unlabeled_ids)

        to_train = math.floor(n_unlabeled*split) - len(self.train_ids)

        if to_train < 0 or to_train > n_unlabeled:
            print("not enough unlabeled samples to successfully split, try setting"
                  "'overwrite_split_labels' to True")
            to_train = int(np.clip(to_train, 0, n_unlabeled)[0])

        np.random.shuffle(unlabeled_ids)
        new_trains = unlabeled_ids[:to_train]
        new_vals = unlabeled_ids[to_train:]

        self.train_ids.extend(new_trains)
        self.val_ids.extend(new_vals)

        for i in new_trains:
            self.examples[i].split = "train"

        for i in new_vals:
            self.examples[i].split = "val"

        sampling.write_json_examples(train_dir + "example.json", self.examples)

