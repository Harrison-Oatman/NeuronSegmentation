from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class Rna:
    y: float
    x: float
    z: float

    barcode: int

    id: int

    def __str__(self):
        return f'({self.y},{self.x})'


class Tester:
    def test(self, rna):
        return True


class Rect:
    def __init__(self, bbox):
        self.y = bbox[0]
        self.x = bbox[1]
        self.h = bbox[2]
        self.w = bbox[3]

        self.ymax = self.y + self.h
        self.xmax = self.x + self.w

    def has_point(self, point):
        return self.y <= point.y < self.ymax and self.x <= point.x < self.xmax

    def intersects(self, other):
        return not (
            self.y > other.ymax or
            other.y > self.ymax or
            self.x > other.xmax or
            other.x > self.xmax
        )

    def contains(self, other):
        return (self.y <= other.y and
                self.x <= other.x and
                self.ymax >= other.ymax and
                self.xmax >= other.xmax)


class Qtree:

    def __init__(self, bbox, limit):
        self.bbox = bbox
        self.rect = Rect(bbox)

        self.count = 0
        self.limit = limit

        self.rna = []
        self.children = None

        self.divided = False

    def add_rna(self, point: Rna):
        if not self.rect.has_point(point):
            return

        if self.count < self.limit:
            self.rna.append(point)
            self.count += 1
            return

        if not self.divided:
            self.children = []
            for y in [0.0, 1.0]:
                for x in [0.0, 1.0]:
                    bbox = [self.rect.y + y * self.rect.h / 2,
                            self.rect.x + x * self.rect.w / 2,
                            self.rect.h / 2,
                            self.rect.w / 2]

                    self.children.append(Qtree(bbox, self.limit))
            self.divided = True
            # for pt in self.rna:
            #     self.add_rna(pt)
            # self.rna = []

        for child in self.children:
            child.add_rna(point)

        self.count += 1

    def __str__(self, depth=1):
        if self.count > 100:
            return f'qtree with {self.count} RNA'
        s = ""
        s += "{"
        s += f"{self.bbox}"
        for rna in self.rna:
            s += f'({str(rna.y)}, {str(rna.x)})'
        if self.children is not None:
            s += '\n'
            for i, child in enumerate(self.children):
                if i > 0:
                    s += '\n'
                s += '\t' * depth
                s += child.__str__(depth + 1)
            s += '\n'
            s += '\t' * (depth - 1)
        s += '}'

        return s

    def query(self, bbox=None, tester=None):
        found = []
        self.get(bbox, tester, found)
        return found

    def get(self, bbox=None, tester=None, found=[]):
        if bbox is None:
            bbox = self.rect

        if not self.rect.intersects(bbox):
            return None

        for item in self.rna:
            if bbox.has_point(item):
                if tester is None:
                    found.append(item)
                else:
                    if item.test(tester):
                        found.append(item)

        if self.divided:
            for child in self.children:
                child.get(bbox, tester, found)

        return None

    def point_plot(self, ax, bbox=None, c='r', s=1, tester=None):
        if bbox is None:
            bbox = self.rect

        points = self.query(bbox, tester)
        ys = [point.y - bbox.y for point in points]
        xs = [point.x - bbox.x for point in points]
        ax.scatter(xs, ys, c=c, s=s)

    def draw(self, ax, bbox=None, c='k', lw=1):
        if bbox is None:
            bbox = self.rect

        if not bbox.intersects(self.rect):
            return

        if bbox.contains(self.rect):
            y1 = self.rect.y - bbox.y
            y2 = self.rect.ymax - bbox.y
            x1 = self.rect.x - bbox.x
            x2 = self.rect.xmax - bbox.x

            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw)

        if self.divided:
            for child in self.children:
                child.draw(ax, bbox, c, lw)

    def circle_query(self, bbox, center, radius2, found, tester=None):
        if not self.rect.intersects(bbox):
            return

        for point in self.rna:
            if (point.y - center[0])**2 + (point.x - center[1])**2 < radius2:
                if tester is None:
                    found.append(point)
                else:
                    if tester.test(point):
                        found.append(point)

        if self.divided:
            for child in self.children:
                child.circle_query(bbox, center, radius2, found)

    def radius_query(self, center, radius, tester=None):
        bbox = Rect([center[0]-radius, center[1]-radius, radius*2, radius*2])
        radius2 = radius**2
        found = []
        self.circle_query(bbox, center, radius2, found, tester=tester)
        return found

    def get_nn(self, point, k, max_dist, tester=None, max_iters=10):
        delta = max_dist/4.0
        rad_guess = max_dist/2.0
        for i in range(max_iters):
            neighbors = self.radius_query((point.y, point.x), rad_guess, tester)
            if len(neighbors) > k:
                rad_guess -= delta
            elif len(neighbors) < k:
                rad_guess += delta
            else:
                return neighbors
            delta /= 2
        return neighbors


class SearchMethod(ABC):
    @abstractmethod
    def search(self, pt) -> list:
        return []


class NearestNeighborSearch(SearchMethod):
    def __init__(self, tree: Qtree, k: int, max_dist):
        self.tree = tree
        self.k = k
        self.max_dist = max_dist

    def search(self, pt) -> list:
        return self.tree.get_nn(pt, self.k, self.max_dist)


class AddMethod:
    # @abstractmethod
    def test(self, pt) -> bool:
        return True


class PointExplorer:
    def __init__(self, tree: Qtree):
        self.qtree = tree
        self.found = {}
        self.horizon = []
        self.accepted = []
        self.rejected = []

    def explore_pt(self, pt, search: SearchMethod, add: AddMethod):
        new_pts = search.search(pt)
        for new_pt in new_pts:
            if self.found.get(new_pt.id) is not None:
                continue
            self.found[new_pt.id] = True
            self.horizon.append(new_pt)

    def expand(self, search: SearchMethod, adder: AddMethod):
        while len(self.horizon) > 0:
            pt = self.horizon.pop()
            if not adder.test(pt):
                self.rejected.append(pt)
                continue
            self.explore_pt(pt, search, adder)
            self.accepted.append(pt)

    def start_point(self, pt):
        self.found[pt.id] = True
        self.horizon.append(pt)


def plot_points(ax, points, c='r', s=1, colorizer=None, offset=(0, 0)):
    ys = [point.y - offset[0] for point in points]
    xs = [point.x - offset[1] for point in points]

    if colorizer is not None:
        c = [colorizer.get_color(point) for point in points]

    ax.scatter(xs, ys, c=c, s=s)


def convert_to_qtree(rna_dataframe, bbox, limit=10):
    qtree = Qtree(bbox, limit)
    rna_list = []
    for index, row in tqdm(rna_dataframe.iterrows()):
        z = row['global_z'] if 'global_z' in rna_dataframe.columns else 0
        point = Rna(row['global_y'], row['global_x'], z, row['barcode_id'], index)
        qtree.add_rna(point)
        rna_list.append(point)
    return qtree, rna_list


# qtree = Qtree([0.0, 0.0, 1.0, 1.0], 3)
# rna = [Rna(np.random.rand(), np.random.rand(), 0.0, i, i) for i in range(5000)]
#
# for r in rna:
#     qtree.add_rna(r)

# ax = plt.subplot()
# # qtree.draw(ax)
# qtree.point_plot(ax)
# # center_points = qtree.radius_query([0.5, 0.5], 0.25)
# # nn = qtree.get_nn(rna[0], 500, 0.2)
# # plot_points(ax, nn, c='b')
# # print(len(nn))
# group = PointExplorer(qtree)
# searcher = NearestNeighborSearch(qtree, 10, 0.02)
# adder = AddMethod()
# group.start_point(rna[0])
# group.expand(searcher, adder)
# plot_points(ax, group.accepted, c='b')
# plt.show()
