import numpy as np
import pandas as pd

from .. import const
from ..utils import cropping
from .inferencesamples import InferenceSample
from ..model import movement
import json
import cv2


def inference_sample_from_json(json_file):
    with open(json_file) as f:
        example_dict = json.loads(f.read())

    inference_sample = InferenceSample(
        input=example_dict["input"],
        centers=example_dict["centers"],
        source=example_dict["source"],
        label=example_dict["label"]
    )

    return inference_sample


class InferenceBatch:

    def __init__(self, image, center, window_size=const.WINDOW_SIZE):
        self.image = image
        self.center = np.array((center["pt_y"], center["pt_x"]))
        self.window_shape = np.array((window_size, window_size))
        self.movequeue = None
        self.valid_batch = True
        self.first_pass_data = self._first_pass()

        # test if the start location was valid
        if self.first_pass_data is None:
            self.valid_batch = False

    """
    This first_pass organization is used to catch bad start locations
    """
    def _first_pass(self):
        cropped_inputs = cropping.batch_crop(self.image, self.center, self.window_shape)
        return cropped_inputs

    def first_pass(self):
        return self.first_pass_data

    def initialize_with_queue(self, movequeue: movement.MoveQueue):
        self.movequeue = movequeue

    def __iter__(self):
        if self.movequeue is None:
            print("batch iteration beginning without movequeue")
        return self

    def __next__(self):
        searching = True
        offset = None
        image = None

        while searching:
            offset = self.movequeue.get_next_loc()
            if offset is None:
                raise StopIteration

            image = cropping.batch_crop(self.image, self.center+offset, self.window_shape)

            if image is not None:
                searching = False

        return image, np.array([offset])


class InferenceLoader:

    def __init__(self, json_file, shuffle=False):

        inference_sample = inference_sample_from_json(json_file)

        self.centers = pd.read_csv(inference_sample.centers)
        self.image = np.expand_dims(np.array(np.load(inference_sample.input))/255., 0)

        self.ids = np.arange(self.centers.shape[0])
        self.i = None

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.ids)
        self.i = 0
        return self

    def __next__(self) -> InferenceBatch:
        searching_for_batch = True
        batch = None
        while searching_for_batch:
            if self.i >= self.centers.shape[0]:
                raise StopIteration

            print("start", self.ids[self.i])
            print("stop")

            batch = InferenceBatch(self.image, self.centers.iloc[self.ids[self.i]])
            self.i += 1
            if batch.valid_batch:
                searching_for_batch = False
        return batch
