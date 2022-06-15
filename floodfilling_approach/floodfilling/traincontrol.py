from . import const
from .training.datasplitter import Splitter
from .training.dataloaders import Dataloader
from .model.ffn import FFN
from tqdm import tqdm
import tensorflow as tf
import time

from .utils.logging import Logger


class TrainController:

    def __init__(self, train_dir=const.TRAINING_DIR, tval_split=const.TRAIN_VAL_SPLIT,
                 batch_size_train=const.BATCH_SIZE_TRAIN, batch_size_val=const.BATCH_SIZE_VAL,
                 first_step_grad=False, net_path=const.NET_PATH, log_path=const.LOG_PATH):

        self.splitter = Splitter(train_dir=train_dir, split=tval_split,
                                 overwrite_split_labels=False)
        self.train_loader = Dataloader("train", self.splitter, batch_size=batch_size_train)
        self.val_loader = Dataloader("val", self.splitter, batch_size=batch_size_val)
        self.first_step_grad = first_step_grad

        self.logger = None
        self.log_path = log_path
        self.log_writer = None

        self.model = None
        self.net_path = f'{net_path}{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}\\'
        self.last_saved = None

        self.recent_batch = None

    def _grad_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self.model.net(inputs, training=True)
            loss = self.model.loss_fn(labels, logits)

        grads = tape.gradient(loss, self.model.net.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.net.trainable_weights))
        return logits, loss

    def train(self, model: FFN, epochs=200):

        self.model = model
        self.start_logging()

        for epoch in (pbar := tqdm(range(epochs))):

            self.logger.start_epoch(epoch)

            self.train_epoch(epoch, pbar)
            self.val_epoch(epoch, pbar)

            self.logger.end_epoch()

        model.net.save(self.net_path+"final")

        # fig, axes = plt.subplots(4, 8)
        # plt.axis('off')
        # print(batch.offsets)
        # for j in range(4):
        #     axes[j, 0].imshow(inputs_a[j, :, :, 0:3])
        #     axes[j, 1].imshow(inputs_a[j, :, :, 3])
        #     axes[j, 2].imshow(labels_a[j])
        #     axes[j, 3].imshow(logits_a[j])
        #     axes[j, 4].imshow(inputs_b[j, :, :, 0:3])
        #     axes[j, 5].imshow(inputs_b[j, :, :, 3])
        #     axes[j, 6].imshow(labels_b[j])
        #     axes[j, 7].imshow(logits_b[j])
        # plt.show()

    def end_epoch(self, epoch):
        self.logger.end_epoch()
        if time.time() - self.last_saved > 600:
            print("saving model at epoch")
            self.model.net.save(f"{self.net_path}{epoch}\\")

    def train_epoch(self, epoch, pbar: tqdm):
        for i, batch in enumerate(self.train_loader):
            pbar.set_description(f"epoch {epoch} train {i}")

            # run new batch procedure on model
            self.model.start_training_batch()

            # get input data
            inputs_a, labels_a = batch.first_pass()
            inputs_a = tf.constant(self.model.pom.start_batch(inputs_a / 255.))

            # first inference
            logits_a = self.model.net(inputs_a, training=False)
            loss_a = None

            if self.first_step_grad:  # shouldn't use?
                logits_a, loss_a = self._grad_step(inputs_a, labels_a)

            # update pom and calculate new offsets
            self.model.apply_inference(logits_a)

            # log loss, accuracy
            self.logger.log("train_a", loss=loss_a, logits=logits_a, labels=labels_a)

            # get new input and labels based on offsets
            inputs_b, labels_b = batch.second_pass(self.model.movequeue)
            inputs_b = tf.constant(self.model.pom.request_poms(inputs_b / 255., batch.offsets))

            # model step
            logits_b, loss_b = self._grad_step(inputs_b, labels_b)

            # log loss, accuracy
            self.logger.log("train_b", loss=loss_b, logits=logits_b, labels=labels_b)

    def val_epoch(self, epoch, pbar: tqdm):

        for i, batch in enumerate(self.val_loader):
            pbar.set_description(f"epoch {epoch} validation {i}")

            # run new batch procedure on model
            self.model.start_training_batch()

            # get input data
            inputs_a, labels_a = batch.first_pass()
            inputs_a = tf.constant(self.model.pom.start_batch(inputs_a / 255.))

            # first inference
            logits_a = self.model.net(inputs_a, training=False)

            # update pom and calculate new offsets
            self.model.apply_inference(logits_a)

            # get new input and labels based on offsets
            inputs_b, labels_b = batch.second_pass(self.model.movequeue)
            inputs_b = tf.constant(self.model.pom.request_poms(inputs_b / 255., batch.offsets))

            # model step
            logits_b = self.model.net(inputs_b, training=False)
            loss_b = self.model.loss_fn(labels_b, logits_b)

            # log loss, acc
            self.logger.log("val_b", loss=loss_b, logits=logits_b, labels=labels_b)
            self.logger.image("val_second_step", inputs_b, labels_b, logits_b)

        self.recent_batch = batch

    def start_logging(self):
        log_dir = self.log_path + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_writer = tf.summary.create_file_writer(log_dir)

        print(f"creating file writer at {log_dir}")
        self.logger = Logger(log_writer, self.model)

        self.last_saved = time.time()
