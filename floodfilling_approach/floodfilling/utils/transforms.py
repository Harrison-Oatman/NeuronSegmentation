import tensorflow as tf
import tensorflow_transform as tft


class Transforms:

    def __init__(self):
        pass

    def preprocess(self, inputs, label):
        inputs = inputs.copy()
        inputs[..., 0] /= 255.
        input_out = tf.constant(inputs, dtype=tf.float32)
        label_out = tf.constant(label, dtype=tf.float32)

        return input_out, label_out
