import math
import numpy as np
from ..import const
from queue import PriorityQueue


class MoveQueue:

    def __init__(self, valid_moves, directions, window_size=const.WINDOW_SIZE,
                 delta=const.DELTA_MAX, threshold=const.VISIT_THRESHOLD):
        self.window_shape = np.array((window_size, window_size), dtype=int)
        self.delta = delta
        self.valid_moves = valid_moves
        self.directions = directions

        self.logit_threshold = math.log(threshold/(1 - threshold))

        self.location = np.array((0, 0), dtype=int)
        self.visited = {(0, 0)}
        self.to_visit = PriorityQueue()

    def register_visit(self, inference):
        """
        send inference from most recently visited location to add
        future locations to the queue

        args:
            inference: (1, w_s, w_s, 1)
        """
        centers = self.window_shape // 2
        checks = self.valid_moves + centers

        vals = inference[0, ..., 0]
        vals = np.take(vals, checks[..., 0]*vals.shape[0] + checks[..., 1])

        max_vals = np.max(vals, axis=-1)
        go_to = np.argsort(max_vals)
        for i in go_to:

            # skip if direction does not reach threshold
            val = max_vals[i]
            if val < self.logit_threshold:
                continue

            # skip if already visited
            true_loc = self.location + self.directions[i]
            if tuple(true_loc) in self.visited:
                continue

            self.to_visit.put((-val, tuple(true_loc)))

    def get_next_loc(self):
        """
        gets the next location to visit
        """
        visited_loc = True
        next_loc = None

        while visited_loc:
            if self.to_visit.empty():
                return None

            _, next_loc = self.to_visit.get()
            visited_loc = next_loc in self.visited

        self.location = next_loc
        self.visited.add(next_loc)

        return tuple(off*self.delta for off in next_loc)


class BatchMoveQueue:

    def __init__(self, valid_moves, directions, window_size=const.WINDOW_SIZE,
                 delta=const.DELTA_MAX, threshold=const.VISIT_THRESHOLD):
        self.window_size = window_size
        self.delta = delta
        self.valid_moves = valid_moves
        self.directions = directions
        self.threshold = threshold

        self.n_children = None
        self.movequeues = None

    def register_visit(self, inference):
        if self.movequeues is None:
            self.n_children = inference.shape[0]
            self.movequeues = [MoveQueue(self.valid_moves, self.directions, self.window_size,
                                         self.delta, self.threshold)
                               for i in range(self.n_children)]


        for i in range(self.n_children):
            self.movequeues[i].register_visit(inference[i:i+1])
