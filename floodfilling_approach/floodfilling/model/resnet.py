from keras.layers import Layer, Conv2D, Add, ReLU, BatchNormalizationV2
from keras import Model
import tensorflow as tf


class ResBlock(Layer):

    def __init__(self, n_filters, layer_name="i", k=5):
        super(ResBlock, self).__init__()

        self.relu = ReLU(name=f"relu_{layer_name}")
        self.conv_a = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                             activation="relu", name=f"resconv{layer_name}_a")

        self.conv_b = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                             activation=None, name=f"resconv{layer_name}_b")
        self.add = Add(name=f"add_{layer_name}")

    def call(self, inputs, *args, **kwargs):
        x = self.relu(inputs)
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.add([x, inputs])
        return x


class ConvStack2DFFN(Model):

    def __init__(self, input_shape, n_filters=32, depth=5, k=5):
        super(ConvStack2DFFN, self).__init__()

        self.depth = depth

        self.batch_norm = BatchNormalizationV2()
        self.conv_0_a = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                               activation="relu", name="conv_0_a",
                               input_shape=input_shape)
        self.conv_0_b = Conv2D(n_filters, kernel_size=(k, k), padding="same",
                               activation=None, name="conv_0_a")

        self.res_blocks = [ResBlock(n_filters, layer_name=str(i))
                           for i in range(depth)]

        self.relu = ReLU(name="relu_final")
        self.out_layer = Conv2D(1, kernel_size=(1, 1), padding="same",
                                activation=None, name="logit_out")

    @tf.function
    def call(self, inputs, training=None, mask=None, depth=None):
        if depth is None:
            depth = self.depth

        x = self.batch_norm(inputs)
        x = self.conv_0_a(x)
        x = self.conv_0_b(x)
        for i in range(min(self.depth, depth)):
            x = self.res_blocks[i](x)
        x = self.relu(x)
        return self.out_layer(x)

    def graph_trace(self, input_patch):
        self.call(input_patch)