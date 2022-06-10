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

        examples = example.load_json_examples()