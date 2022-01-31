from abc import ABC, abstractmethod
from trees import Qtree

"""
This file contains objects and methods which are used to generate groups
of RNA by some method of addition
"""


class SearchMethod(ABC):
    """
    A SearchMethod adds new RNA to the horizon
    """
    @abstractmethod
    def search(self, pt) -> list:
        return []


class NearestNeighborSearch(SearchMethod):
    """
    This simple search finds the nearest neighbors to a point on a Tree
    """
    def __init__(self, tree: Qtree, k: int, max_dist):
        self.tree = tree
        self.k = k
        self.max_dist = max_dist

    def search(self, pt) -> list:
        return self.tree.get_nn(pt, self.k, self.max_dist)


class AddMethod:
    """
    An AddMethod determines if a point on the horizon should be accepted
    """
    # @abstractmethod
    def test(self, pt) -> bool:
        return True


class PointExplorer:
    """
    A point explorer uses a search method to build a horizon,
    and a test method to accept from the horizon
    """
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
