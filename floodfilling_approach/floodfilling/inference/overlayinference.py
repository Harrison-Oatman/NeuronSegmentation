import numpy as np
import cmapy
import random
import matplotlib.pyplot as plt
from ..utils import cropping


class GlobalInference:
    """
    A class for storing an image representing inferences made my the flood filling network.
    This is intended to produce a figure, not apply segmentation
    """
    def __init__(self, bg_image, cmap='gist_rainbow'):
        # convert input layer to rgb
        self.global_image = np.dstack([bg_image[0, ..., 0] for _ in range(3)])
        self.global_image = self.global_image / max(np.max(self.global_image), 1.0)
        self.colormap = cmap
        self.overlay_opacity = 0.8

    def _random_color(self) -> tuple[float]:
        rgb_color = cmapy.color(self.colormap, random.randrange(0, 256), rgb_order=True)
        return tuple(float(rgb_color[i])/255. for i in range(3))

    def _norm_and_color(self, image):
        mask = image == 0.0
        normed = 1/(1 + np.exp(-image))
        normed[mask] = 0.0
        color = self._random_color()
        colored = np.dstack([normed*a for a in color])
        return colored, normed

    def overlay(self, image_two, center):
        """
        overlays image_two onto image_one with the center of image_two
        at 'center'. image
        """
        colored, normed = self._norm_and_color(np.squeeze(image_two))

        a_slice, b_slice = cropping.imalign(self.global_image, colored, center)
        a = self.global_image[a_slice]

        b = colored[b_slice]
        c = np.expand_dims(normed[b_slice] * self.overlay_opacity, axis=-1)
        overlay = np.multiply(a, (1-c)) + np.multiply(b, c)

        self.global_image[a_slice] = overlay
