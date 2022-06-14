from . import const
from .training.datasplitter import Splitter
from .training.dataloaders import Dataloader
from .model.ffn import FFN
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


class TrainController:

    def __init__(self, train_dir=const.TRAINING_DIR, tval_split=const.TRAIN_VAL_SPLIT,
                 batch_size_train=const.BATCH_SIZE_TRAIN, batch_size_val=const.BATCH_SIZE_VAL,
                 first_step_grad=False):

        self.splitter = Splitter(train_dir=train_dir, split=tval_split,
                                 overwrite_split_labels=False)
        self.train_loader = Dataloader("train", self.splitter, batch_size=batch_size_train)
        self.val_loader = Dataloader("val", self.splitter, batch_size=batch_size_val)
        self.first_step_grad = first_step_grad

    def train(self, model: FFN, epochs=200):

        def _grad_step(inputs, labels):
            with tf.GradientTape() as tape:
                logits = model.net(inputs, training=True)
                loss = model.loss_fn(labels, logits)

            grads = tape.gradient(loss, model.net.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.net.trainable_weights))
            return logits, loss

        for epoch in range(epochs):
            train_loss = 0
            train_correct = 0
            train_total = 0

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"train: epoch {epoch}"):

                # run new batch procedure on model
                model.start_training_batch()

                # get input data
                inputs_a, labels_a = batch.first_pass()
                inputs_a = tf.constant(model.pom.start_batch(inputs_a / 255.))

                # first inference
                logits_a = model.net(inputs_a, training=False)

                if self.first_step_grad:  # shouldn't use?
                    logits_a, loss_a = _grad_step(inputs_a, labels_a)

                # update pom and calculate new offsets
                model.apply_inference(logits_a)

                # get new input and labels based on offsets
                inputs_b, labels_b = batch.second_pass(model.movequeue)
                inputs_b = tf.constant(model.pom.request_poms(inputs_b / 255., batch.offsets))

                # model step
                logits_b, loss_b = _grad_step(inputs_b, labels_b)

                # log loss, accuracy
                train_loss += tf.reduce_sum(loss_b)
                train_correct += model.calc_accuracy(logits_b, labels_b)
                train_total += 1

            print(f"train loss: {train_loss.numpy()}")
            print(f"train acc: {train_correct / train_total}")

            val_loss = 0
            val_correct = 0
            val_total = 0

            for i, batch in tqdm(enumerate(self.val_loader), desc=f"val: epoch {epoch}"):
                # run new batch procedure on model
                model.start_training_batch()

                # get input data
                inputs_a, labels_a = batch.first_pass()
                inputs_a = tf.constant(model.pom.start_batch(inputs_a/255.))

                # first inference
                logits_a = model.net(inputs_a, training=False)

                # update pom and calculate new offsets
                offsets = model.apply_inference(logits_a)

                # get new input and labels based on offsets
                inputs_b, labels_b = batch.second_pass(model.movequeue)
                inputs_b = tf.constant(model.pom.request_poms(inputs_b/255., batch.offsets))

                # model step
                logits_b = model.net(inputs_b, training=False)
                loss_b = model.loss_fn(labels_b, logits_b)

                # log loss, accuracy
                val_loss += tf.reduce_sum(loss_b)
                val_correct += model.calc_accuracy(logits_b, labels_b)
                val_total += 1

            print(f"val loss: {val_loss.numpy()}")
            print(f"val acc: {val_correct/val_total}")

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
