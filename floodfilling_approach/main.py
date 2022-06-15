from floodfilling.training import dataloaders
from floodfilling.model import ffn
import matplotlib.pyplot as plt
import numpy as np
from floodfilling.model.resnet import ConvStack2DFFN
from floodfilling.traincontrol import TrainController
import tensorflow as tf
import keras
import datetime
from floodfilling.utils import sampling
from floodfilling.utils import foldercleaning
from floodfilling.inferencecontroller import InferenceController

# foldercleaning.main()
# foldercleaning.main("0520")

# sampling.main()

# splitter = dataloaders.Splitter()
# train_loader = dataloaders.Dataloader("train", splitter)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)

net = ConvStack2DFFN(input_shape=(67, 67, 3), depth=6, k=7)
model = ffn.FFN()
model.net = net
controller = TrainController(first_step_grad=True)
#
controller.train(model, epochs=200)
# # #
# net_path = "C:\\Lab Work\\segmentation\\saved_models\\wideffnmodel\\process_set\\"
# model.net.save(net_path)
#
# net = keras.models.load_model(net_path)
#
# model = ffn.FFN()
# model.net = net
#
# controller = InferenceController()
# controller.inference(model)


# tf.summary.trace_on(graph=True, profiler=True)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# for i in range(20):
#     for batch in train_loader:
#         inputs, labels = batch.first_pass()
#         labels = np.array(labels, dtype=np.float32)
#         inputs = np.array(inputs, dtype=np.float32)
#         inputs /= 255.
#         return_dict = model.train_on_batch(inputs, labels, return_dict=True)
#     print(return_dict)


# @tf.function
# def trace_me(x):
#     return model.call(x)
#
#
# plt.imshow(net(tf.random.normal((1,67,67,4)))[0])
# plt.show()
# with writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)


# fig, axes = plt.subplots(3, 3)
# for i in range(3):
#     axes[i, 0].imshow(inputs[i])
#     axes[i, 1].imshow(model(inputs[i:i+1])[0])
#     axes[i, 2].imshow(labels[i])
# plt.show()
