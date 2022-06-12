import tensorflow as tf
from keras.layers import Layer, Conv2D, Add, ReLU
from keras import Model


class ResBlock(Layer):

    def __init__(self, n_filters, layer_name="i"):
        super(ResBlock, self).__init__()

        self.relu = ReLU(name=f"relu_{layer_name}")
        self.conv_a = Conv2D(n_filters, kernel_size=(5, 5), padding="same",
                             activation="relu", name=f"resconv{layer_name}_a")

        self.conv_b = Conv2D(n_filters, kernel_size=(5, 5), padding="same",
                             activation=None, name=f"resconv{layer_name}_b")
        self.add = Add(name=f"add_{layer_name}")

    def call(self, inputs, *args, **kwargs):
        x = self.relu(inputs)
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.add([x, inputs])
        return x


class ConvStack2DFFN(Model):

    def __init__(self, input_shape, n_filters=32, depth=5):
        super(ConvStack2DFFN, self).__init__()

        self.depth = depth

        self.conv_0_a = Conv2D(n_filters, kernel_size=(5, 5), padding="same",
                               activation="relu", name="conv_0_a",
                               input_shape=input_shape)
        self.conv_0_b = Conv2D(n_filters, kernel_size=(5, 5), padding="same",
                               activation=None, name="conv_0_a")

        self.res_blocks = [ResBlock(n_filters, layer_name=str(i))
                           for i in range(depth)]

        self.relu = ReLU(name="relu_final")
        self.out_layer = Conv2D(1, kernel_size=(1,1), padding="same",
                                activation=None, name="logit_out")

    def call(self, inputs, training=None, mask=None):
        x = self.conv_0_a(inputs)
        x = self.conv_0_b(x)
        for i in range(self.depth):
            x = self.res_blocks[i](x)
        x = self.relu(x)
        return self.out_layer(x)


