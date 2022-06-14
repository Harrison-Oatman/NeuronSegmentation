from .resnet import ConvStack2DFFN
from ..training import optimizer
import tensorflow as tf
from .. import const
import numpy as np
from .pom import POM
from .movement import BatchMoveQueue, MoveQueue

class FFN:

    def __init__(self, delta_max=const.DELTA_MAX):

        self.net = None
        self.optimizer = None
        self.loss_fn = None

        self.pom = POM()

        self.delta_max = delta_max

        self.set_up_optimizer()
        self.set_up_loss()
        self.moves = self.valid_modes()

        self.accuracy = 0

        self.movequeue = None

    def set_up_optimizer(self):
        self.optimizer = optimizer.optimizer_from_config()

    def set_up_loss(self):
        self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

    def start_inference_batch(self):
        valid_moves, directions = self.valid_modes()
        self.movequeue = MoveQueue(valid_moves, directions)

    def start_training_batch(self):
        valid_moves, directions = self.valid_modes()
        self.movequeue = BatchMoveQueue(valid_moves, directions, threshold=0.00000001)

    def apply_inference(self, inference, inference_step=False):
        self.pom.update_poms(inference, inference_step)
        self.movequeue.register_visit(inference)

    def calc_accuracy(self, inference, labels):
        self.accuracy = 1 - np.average(np.bitwise_xor(inference > 0, labels > 0.5))
        return self.accuracy

    def valid_modes(self):
        ys = []
        xs = []

        for cs, bs in [(xs, ys), (ys, xs)]:
            for c in [-self.delta_max, self.delta_max]:
                cs.append([])
                bs.append([])
                for b in np.arange(start=-self.delta_max+1, stop=self.delta_max):
                    cs[-1].append(c)
                    bs[-1].append(b)

        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        return np.array(np.dstack([ys, xs]), dtype=int), np.array(directions, dtype=int)
