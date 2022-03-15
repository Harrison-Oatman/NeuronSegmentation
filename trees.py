from basicrna import Rna, Colorizer, plot_points
from tqdm import tqdm


class Rect:
    """
    A rect is a rectangle, and it can make some simple comparisons.
    It is used primarily by Qtree
    """
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


"""
Testers are ways to specify a search
"""


class Tester:
    def test(self, rna):
        return True


class BarcodeTester(Tester):
    def __init__(self, barcode_id):
        self.id = barcode_id

    def test(self, rna):
        return rna.barcode == self.id


"""
Getters can be passed into searches in order to collect data while a search is ongoing
"""


class Getter:
    def get(self, pt):
        pass

    def clear(self):
        pass


class RCVGetter(Getter):
    """
    Gets an RCV during a search
    """
    def __init__(self, rcv):
        self.rcv = rcv

    def get(self, pt):
        self.rcv[pt.barcode] += 1.0

    def clear(self):
        self.rcv[:] = 0


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

    def get(self, bbox, tester, found):

        if not self.rect.intersects(bbox):
            return None

        for item in self.rna:
            if bbox.has_point(item):
                if tester is None:
                    found.append(item)
                else:
                    if tester.test(item):
                        found.append(item)

        if self.divided:
            for child in self.children:
                child.get(bbox, tester, found)

        return None

    def plot_points(self, ax, bbox=None, c='r', s=1, tester=None, colorizer: Colorizer = None, image=None):
        if bbox is None:
            bbox = self.rect

        points = self.query(bbox, tester)
        plot_points(ax, points, c=c, s=s, colorizer=colorizer, image=None)

    def draw(self, ax, bbox=None, c='k', lw=1):
        if bbox is None:
            bbox = self.rect

        if not bbox.contains(self.rect):
            return

        y1 = self.rect.y - bbox.y
        y2 = self.rect.ymax - bbox.y
        x1 = self.rect.x - bbox.x
        x2 = self.rect.xmax - bbox.x

        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw)

        if self.divided:
            for child in self.children:
                child.draw(ax, bbox, c, lw)

    def circle_query(self, bbox, center, radius2, found, tester=None, getter=None):
        if not self.rect.intersects(bbox):
            return

        for point in self.rna:
            if (point.y - center[0])**2 + (point.x - center[1])**2 < radius2:
                if tester is None:
                    found.append(point)
                    getter.get(point)
                else:
                    if tester.test(point):
                        found.append(point)
                        getter.get(point)

        if self.divided:
            for child in self.children:
                child.circle_query(bbox, center, radius2, found, tester, getter)

    def radius_query(self, center, radius, tester=None, getter=None):
        if getter is None:
            getter = Getter()
        bbox = Rect([center[0]-radius, center[1]-radius, radius*2, radius*2])
        radius2 = radius**2
        found = []
        self.circle_query(bbox, center, radius2, found, tester=tester, getter=getter)
        return found

    def get_nn(self, point, k, max_dist, tester=None, max_iters=10, rcv=None, id=None):
        delta = max_dist/4.0
        rad_guess = max_dist/2.0

        # rcv specific code
        if rcv is not None:
            getter = RCVGetter(rcv[id, :])
            # print(getter.rcv)
        else:
            getter = Getter()

        for i in range(max_iters):
            getter.clear()
            neighbors = self.radius_query((point.y, point.x), rad_guess, tester, getter)
            if len(neighbors) > k:
                rad_guess -= delta
            elif len(neighbors) < k:
                rad_guess += delta
            else:
                return neighbors
            delta /= 2
        return neighbors


def convert_to_qtree(rna_dataframe, bbox, limit=10):
    qtree = Qtree(bbox, limit)
    rna_list = []
    global_z = True if 'global_z' in rna_dataframe.columns else False
    cell_index = True if 'cell_index' in rna_dataframe.columns else False
    soma_distance = True if 'distance2Center' in rna_dataframe.columns else False

    for index, row in tqdm(rna_dataframe.iterrows()):
        z = row['global_z'] if global_z else 0
        cell = row['cell_index'] if cell_index else 0
        distance = row['distance2Center'] if soma_distance else 0
        point = Rna(row['global_y'], row['global_x'], z, int(row['barcode_id']),
                    index, cell=cell, processIndex=row['process_index'], somaDistance=distance)
        qtree.add_rna(point)
        rna_list.append(point)
    return qtree, rna_list


def generate_barcode_qtrees(rna_list, bbox, n_rna=1240, limit=10):
    barcode_qtrees = {i: Qtree(bbox, limit) for i in range(n_rna)}
    barcode_rna_lists = {i: [] for i in range(n_rna)}
    for i, point in tqdm(enumerate(rna_list)):
        barcode_qtrees[point.barcode].add_rna(point)
        barcode_rna_lists[point.barcode].append(point)

    return barcode_qtrees, barcode_rna_lists
