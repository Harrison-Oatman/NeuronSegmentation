from .resnet import ConvStack2DFFN
from ..training import optimizer
import tensorflow as tf
from .. import const
import numpy as np
from .pom import POM

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

    def set_up_optimizer(self):
        self.optimizer = optimizer.optimizer_from_config()

    def set_up_loss(self):
        self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

    def apply_inference(self, inference):
        self.pom.update_poms(inference)

        # todo: implement move calculation
        next_move = [[12, 0] for _ in inference]

        return next_move

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

        return np.array(np.dstack([ys, xs]))
