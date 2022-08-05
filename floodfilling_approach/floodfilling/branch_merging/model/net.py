from keras.layers import Layer, Conv2D, Add, \
    ReLU, BatchNormalizationV2, GlobalAvgPool2D, Dense
from keras import Model
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from ...const import *


# def get_resnet():
#     net = ResNet50V2(
#         include_top=False,
#         input_shape=(BRANCH_WINDOW_SIZE, BRANCH_WINDOW_SIZE, 3),
#         pooling='avg',
#     )
#
#     return net


class ResBlock(Layer):

    def __init__(self, n_filters, layer_name="i", k=5):
        super(ResBlock, self).__init__()

        batch_norm = BatchNormalizationV2

        self.conv_a = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                             activation=None, name=f"resconv_{layer_name}_a")
        self.batch_norm_a = batch_norm(name=f"batchnorm_{layer_name}_a")
        self.relu_a = ReLU(name=f"relu_{layer_name}_a")

        self.conv_b = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                             activation=None, name=f"resconv_{layer_name}_b")
        self.batch_norm_b = batch_norm(name=f"batchnorm_{layer_name}_b")
        self.add = Add(name=f"add_{layer_name}")

        self.relu_out = ReLU(name=f"relu_{layer_name}_out")

    def call(self, inputs, *args, **kwargs):
        x = self.conv_a(inputs)
        x = self.batch_norm_a(x)
        x = self.relu_a(x)
        x = self.conv_b(x)
        x = self.batch_norm_b(x)
        x = self.add([x, inputs])
        return x


class ResNet(Model):

    def __init__(self, input_shape, n_filters=32, k=5):
        super(ResNet, self).__init__()

        self.batch_norm = BatchNormalizationV2()
        self.conv_0_a = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                               activation="relu", name="conv_0_a",
                               input_shape=input_shape)
        self.conv_0_b = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                               activation=None, name="conv_0_a")

        self.block_a = ResBlock(n_filters, "1")
        self.block_b = ResBlock(n_filters, "2")
        self.block_c = ResBlock(n_filters, "3")

        self.relu = ReLU(name="relu_final")
        self.out_layer = Conv2D(1, kernel_size=(1, 1), padding="same",
                                activation="sigmoid", name="logit_out")

        self.pooling = GlobalAvgPool2D()

    @tf.function
    def call(self, inputs, training=None, mask=None, depth=None):

        x = self.batch_norm(inputs)
        x = self.conv_0_a(x)
        x = self.conv_0_b(x)
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.relu(x)
        x = self.out_layer(x)
        x = self.pooling(x)
        return x

