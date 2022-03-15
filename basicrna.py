import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Rna:
    y: float
    x: float
    z: float
    barcode: int
    id: int  # id which may be used to reference this Rna in other contexts

    somaDistance: float = 0
    cell: int = 0  # the label for the cell_test the rna belongs to
    processIndex: int = 0  # index used to specify process_test within cell_test


class Colorizer(ABC):
    """
    Takes an rna and returns a point
    """
    @abstractmethod
    def get_color(self, pt):
        pass


class ColorByCell(Colorizer):
    def get_color(self, point):
        return point.cell_test


class ColorByBarcode(Colorizer):
    def get_color(self, point):
        return point.barcode


def plot_points(ax, points, c='r', s=1, colorizer: Colorizer = None, offset=(0, 0), image=None, **kwargs):
    """
    This function is used for plotting a list of rna. We can specify a list of colors
    using c, or send in a colorizer with the function get_color which return
    """
    ys = [point.y - offset[0] for point in points]
    xs = [point.x - offset[1] for point in points]

    if colorizer is not None:
        c = [colorizer.get_color(point) for point in points]

    if image is not None:
        ax.imshow(image, **kwargs)

    ax.scatter(xs, ys, c=c, s=s)
