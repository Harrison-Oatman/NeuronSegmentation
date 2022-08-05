import numpy as np
import tensorflow as tf
import tensorflow_transform as tft


class Transforms:

    def __init__(self):
        pass

    def preprocess(self, inputs, label):
        inputs = inputs.copy()
        inputs[..., 0] /= 255.
        label = [0.05 if i == 0 else 0.95 for i in label]
        input_out = tf.constant(inputs, dtype=tf.float32)
        label_out = tf.constant(label, dtype=tf.float32)

        return input_out, label_out


class BranchTransforms:

    def __init__(self):
        pass

    def preprocess(self, inputs, pom, label):
        inputs = inputs.copy()
        inputs[..., 0] /= 255.

        stack = [inputs[..., i] for i in range(inputs.shape[-1])]
        stack.extend([pom[..., i] for i in range(pom.shape[-1])])

        input_out = np.stack(stack, axis=-1)
        input_out = tf.constant(input_out, dtype=tf.float32)
        label_out = tf.constant(label, dtype=tf.float32)

        return input_out, label_out
