import io

import tensorflow as tf
from matplotlib import pyplot as plt


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def grid_plots(inputs, logits, labels, max_inputs=8):

    plt.tight_layout()
    figures = []

    for i in range(min(max_inputs, inputs.shape[0])):
        figure, axes = plt.subplots(1, 4)

        axes[0].imshow(inputs[i, ..., :3])
        axes[0].set_title("input")
        axes[1].imshow(inputs[i, ..., -1:])
        axes[1].set_title("seed")
        axes[2].imshow(labels[i])
        axes[2].set_title("label")
        axes[3].imshow(logits[i])
        axes[3].set_title("output")
        figures.append(figure)

    return figures