from floodfilling.sampling import foldercleaning, sampling
from floodfilling.model.resnet import ConvStack2DFFN
from floodfilling.model import ffn
from floodfilling.traincontrol import TrainController
from floodfilling.inferencecontroller import InferenceController
import tensorflow as tf
from tensorflow import keras

# foldercleaning.main()
# for src in ["0520", "0602", "0605", "0613", "0807", "0812", "0814",
#             "1014", "1028", "1030", "1102", "1107"]:
#     foldercleaning.main(src)
#     sampling.main(src)
#
# foldercleaning.main("0225", False)
# sampling.main("0520", True)

# splitter = dataloaders.Splitter()
# train_loader = dataloaders.Dataloader("train", splitter)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)

# net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-15_13-42-10\\9"
# # model.net.save(net_path)
# #
# net = keras.models.load_model(net_path)

# net = ConvStack2DFFN(input_shape=(67, 67, 3), depth=8, k=5)
# model = ffn.FFN(net)
# model.net = net
# model.compile()
# controller = TrainController(first_step_grad=True)
# # #
# # controller.train(model, epochs=30, depth=4)
# controller.train(model, epochs=200, depth=8)
# # # #

# net_path = "C:\\Lab Work\\segmentation\\saved_models\\most_recent"
net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-21_18-57-08\\final"
net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-06_17-29-08\\final"
# # net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-21_14-49-11\\14"
net = keras.models.load_model(net_path, custom_objects={
    "sigmoid_cross_entropy_with_logits_v2": tf.nn.sigmoid_cross_entropy_with_logits
})
model = ffn.FFN(net, train=False)
# model.net = net
# #
controller = InferenceController()
controller.inference(model)

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
