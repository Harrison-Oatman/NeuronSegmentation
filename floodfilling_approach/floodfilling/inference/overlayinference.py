import numpy as np
import cmapy
import random
import matplotlib.pyplot as plt


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

        y, x = center
        q, p = self.global_image.shape[:2]
        print(image_two.shape)
        m, n, _ = image_two.shape

        aymin = max(y - m//2 - 1, 0)
        aymax = min(y + m//2, q)

        axmin = max(x - n//2 - 1, 0)
        axmax = min(x + n//2, p)

        bymin = max(0, m//2 + 1 - y)
        bymax = min(q - y + m//2 + 1, m)

        bxmin = max(0, n//2 + 1 - x)
        bxmax = min(p - x + n//2 + 1, n)

        a = self.global_image[aymin:aymax, axmin:axmax, :]

        b = colored[bymin:bymax, bxmin:bxmax, :]
        c = np.expand_dims(normed[bymin:bymax, bxmin:bxmax] * self.overlay_opacity, axis=-1)
        overlay = np.multiply(a,(1-c)) + np.multiply(b,c)

        self.global_image[aymin:aymax, axmin:axmax, :] = overlay
