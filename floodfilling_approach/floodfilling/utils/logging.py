import tensorflow as tf
from ..model.ffn import FFN
from . import imaging


class Logger:

    def __init__(self, log_writer:tf.summary.SummaryWriter, model:FFN):
        self.writer = log_writer
        self.model = model

        self.i = None
        self.acc = None
        self.loss = None
        self.global_step = 0

    def start_epoch(self, global_step):
        self.loss = {}
        self.acc = {}
        self.global_step = global_step

    def log(self, name, loss=None, logits=None, labels=None):
        """
        call to add batch-level data to logger
        """
        if loss is not None:
            this_loss = self.loss.get(name, 0)
            self.loss[name] = this_loss + tf.reduce_sum(loss)

        if (logits is not None) and (labels is not None):
            this_acc, i = self.acc.get(name, (0, 0))
            this_acc += self.model.calc_accuracy(logits, labels)
            i += 1
            self.acc[name] = (this_acc, i)

    def image(self, name, sample=None, label=None,
              logit=None, pom=None, max_outputs=8):

        with self.writer.as_default():
            if sample is not None:
                tf.summary.image(f"input_{name}", sample[..., :3], self.global_step, max_outputs)
                tf.summary.image(f"seed_{name}", sample[..., -1:], self.global_step, max_outputs)

            if label is not None:
                tf.summary.image(f"label_{name}", label, self.global_step, max_outputs)

            if logit is not None:
                tf.summary.image(f"output_{name}", logit, self.global_step, max_outputs)

            if pom is not None:
                tf.summary.image(f"POM_{name}", pom, self.global_step, max_outputs)

            if (sample is not None) and (label is not None) and (logit is not None):
                figures = imaging.grid_plots(sample, logit, label)
                for figure in figures:
                    tf.summary.image(f"summary_{name}", imaging.plot_to_image(figure),
                                     self.global_step)

    def end_epoch(self):
        """
        totals and averages losses and accuracy, then writes scalar to tensorboard
        """
        with self.writer.as_default():
            for name in self.loss:
                tf.summary.scalar(f"loss_{name}", self.loss[name], self.global_step)

            for name in self.acc:
                acc_tot, i = self.acc[name]
                acc_mean = acc_tot/i
                tf.summary.scalar(f"acc_{name}", acc_mean, self.global_step)