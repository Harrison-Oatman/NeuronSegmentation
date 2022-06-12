from .datasplitter import Splitter
import numpy as np
from ..utils import cropping
from .. import const


class Batch:

    def __init__(self, samples):
        self.samples = samples

        self.window_size = const.WINDOW_SIZE
        self.window_shape = np.array((self.window_size, self.window_size))

        self.sample_inputs = np.array([np.load(sample.input.image) for sample in self.samples])
        self.sample_labels = np.array([np.load(sample.label.segmentation) for sample in self.samples])

    def first_pass(self):

        cropped_inputs = cropping.crop_offset(self.sample_inputs, np.array((0, 0)), self.window_shape)
        cropped_labels = cropping.crop_offset(self.sample_labels, np.array((0, 0)), self.window_shape)

        return cropped_inputs, cropped_labels

    def second_pass(self, offsets):

        inputs = []
        for i in range(len(self.sample_inputs)):
            inputs.append(cropping.crop_offset(self.sample_inputs[i:i+1, :],
                                               offsets[i],
                                               self.window_shape))
        cropped_inputs = np.vstack(inputs)

        labels = []
        for i in range(len(self.sample_labels)):
            labels.append(cropping.crop_offset(self.sample_labels[i:i + 1, :],
                                               offsets[i, :],
                                               self.window_shape))
        cropped_labels = np.vstack(labels)

        return cropped_inputs, cropped_labels


class Dataloader:

    def __init__(self, split, splitter: Splitter, batch_size=16):
        self.splitter = splitter
        self.samples = self.splitter.get_samples(split)
        self.batch_size = batch_size
        self.i = 0

    def batch(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.samples)
        self.i = 0
        return self

    def __next__(self):
        if self.i + self.batch_size > len(self.samples):
            raise StopIteration

        batch = Batch(self.samples[self.i:self.i + self.batch_size])
        self.i += self.batch_size

        return batch
