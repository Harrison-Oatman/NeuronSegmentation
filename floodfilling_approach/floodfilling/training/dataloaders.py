from .datasplitter import Splitter
import numpy as np
from ..utils import cropping
from .. import const
from ..model import movement


def soften_labels(array):
    a = np.empty(array.shape, dtype=np.float32)
    a.fill(0.05)
    a[array] = 0.95
    return a


class Batch:

    def __init__(self, samples, input_attr="image"):
        self.samples = samples

        self.window_size = const.WINDOW_SIZE
        self.window_shape = np.array((self.window_size, self.window_size))

        self.sample_inputs = np.array([np.load(getattr(sample.input, input_attr)) for sample in self.samples])
        self.sample_labels = np.array([np.load(sample.label.segmentation) for sample in self.samples])
        self.sample_labels = soften_labels(self.sample_labels)

        self.offsets = None

    def first_pass(self):

        cropped_inputs = cropping.crop_offset(self.sample_inputs, np.array((0, 0)), self.window_shape)
        cropped_labels = cropping.crop_offset(self.sample_labels, np.array((0, 0)), self.window_shape)

        return cropped_inputs, cropped_labels

    def second_pass(self, movequeue:movement.BatchMoveQueue):

        offsets = np.array([queue.get_next_loc() for queue in movequeue.movequeues])

        self.offsets = offsets

        cropped_inputs = np.vstack(cropping.batch_crop_offset(self.sample_inputs,
                                                    offsets,
                                                    self.window_shape))

        cropped_labels = np.vstack(cropping.batch_crop_offset(self.sample_labels,
                                                    offsets,
                                                    self.window_shape))

        return cropped_inputs, cropped_labels


class Dataloader:

    def __init__(self, split, splitter: Splitter, batch_size=16, shuffle=True,
                 input_attr=const.INPUT_ATTR):
        self.splitter = splitter
        self.samples = self.splitter.get_samples(split)
        self.batch_size = batch_size
        self.i = 0
        self.shuffle = shuffle
        self.input_attr = input_attr

    def batch(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
        self.i = 0
        return self

    def __next__(self):
        if self.i + self.batch_size > len(self.samples):
            raise StopIteration

        batch = Batch(self.samples[self.i:self.i + self.batch_size], input_attr=self.input_attr)
        self.i += self.batch_size

        return batch
