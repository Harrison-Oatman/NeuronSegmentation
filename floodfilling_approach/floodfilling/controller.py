from . import const
from .training.datasplitter import Splitter
from .training.dataloaders import Dataloader
from .model.ffn import FFN
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


class TrainController:

    def __init__(self, train_dir=const.TRAINING_DIR, tval_split=const.TRAIN_VAL_SPLIT,
                 batch_size_train=const.BATCH_SIZE_TRAIN, batch_size_val=const.BATCH_SIZE_VAL):

        self.splitter = Splitter(train_dir=train_dir, split=tval_split,
                                 overwrite_split_labels=False)
        self.train_loader = Dataloader("train", self.splitter, batch_size=batch_size_train)
        self.val_loader = Dataloader("val", self.splitter, batch_size=batch_size_val)

    def train(self, model: FFN, epochs=20):

        for epoch in range(epochs):
            total_loss = 0

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"epoch {epoch}"):
                inputs_a, labels_a = batch.first_pass()
                inputs_a = model.pom.start_batch(inputs_a/255.)

                logits_a = model.net(inputs_a, training=False)

                # with tf.GradientTape() as tape:
                #     logits_a = model.net(inputs_a, training=True)
                #     loss_a = model.loss_fn(labels_a, logits_a)
                #
                # grads = tape.gradient(loss_a, model.net.trainable_weights)
                # model.optimizer.apply_gradients(zip(grads, model.net.trainable_weights))

                offsets = model.apply_inference(logits_a)
                inputs_b, labels_b = batch.second_pass(offsets)
                inputs_b = model.pom.request_poms(inputs_b/255., offsets)

                with tf.GradientTape() as tape:
                    logits_b = model.net(inputs_b, training=True)
                    loss = model.loss_fn(labels_b, logits_b)

                grads = tape.gradient(loss, model.net.trainable_weights)
                model.optimizer.apply_gradients(zip(grads, model.net.trainable_weights))

                total_loss += tf.reduce_sum(loss)
            print(total_loss)

        fig, axes = plt.subplots(2, 4)
        plt.axis('off')
        axes[0, 0].imshow(inputs_a[0, :, :, 0:3])
        axes[0, 1].imshow(inputs_a[0, :, :, 3])
        axes[0, 2].imshow(labels_a[0])
        axes[0, 3].imshow(logits_a[0])
        axes[1, 0].imshow(inputs_b[0, :, :, 0:3])
        axes[1, 1].imshow(inputs_b[0, :, :, 3])
        axes[1, 2].imshow(labels_b[0])
        axes[1, 3].imshow(logits_b[0])
        plt.show()
