import numpy as np
from .. import const
from ..utils import cropping


class POM:

    def __init__(self, window_size=const.WINDOW_SIZE):
        self.window_size = window_size
        self.window_shape = np.array([window_size, window_size])
        self.poms = None
        self.step = 0
        self.offsets = None

    def start_batch(self, inputs):
        # determine batch shape from inputs
        batch_shape = list(inputs.shape)
        batch_shape[-1] = 1

        self.step = 0

        self.offsets = np.zeros((batch_shape[0], 2), dtype=np.int32)

        # calculate center to produce seed
        centers = list(np.array(batch_shape[1:-1]) // 2)
        centers.append(0)
        centers = tuple(centers)

        # create pom arrays
        pom = np.empty(batch_shape, dtype=np.float32)
        pom.fill(-0.3)
        self.poms = [a for a in pom]
        for pom in self.poms:
            pom[centers] = 0.3

        return np.concatenate([inputs, np.array(self.poms)], axis=-1)

    def request_poms(self, inputs, offsets):
        pom_shapes = np.array([pom.shape[1:-1] for pom in self.poms])
        starts = pom_shapes // 2 - np.array(inputs.shape[1:-1], dtype=int) // 2 + offsets
        ends = starts + np.array(inputs.shape[1:-1], dtype=int)

        start_overlap = np.abs(np.minimum(starts, 0))
        end_overlap = np.abs(np.minimum(pom_shapes-ends, 0))

        for i in range(len(start_overlap)):
            start = start_overlap[i]
            end = end_overlap[i]

            if not (np.all(start == 0) and np.all(end == 0)):
                self.poms[i] = np.pad(self.poms[i],
                                      [(0, 0)] +
                                      [(max(s, e), max(s, e)) for s, e in zip(start, end)] +
                                      [(0, 0)])

        poms = np.vstack([cropping.crop_offset(pom, offsets[i], self.window_shape)
                          for i, pom in enumerate(self.poms)])

        self.offsets = offsets

        return np.concatenate([inputs, poms], axis=-1)

    def update_poms(self, inference):

        old_poms = np.vstack(cropping.batch_crop_offset(self.poms, self.offsets, self.window_shape)).copy()

        freeze = np.zeros(old_poms.shape, dtype=bool)

        if self.step > 0:
            freeze = np.bitwise_and((old_poms < 0.5), (inference > old_poms))

        new_patch = inference.numpy()
        new_patch[freeze] = old_poms[freeze]

        self.poms = cropping.batch_paste_offset(self.poms, self.offsets, new_patch)
