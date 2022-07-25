import numpy as np
import pandas
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from .. import const


def get_random_seeds(regions_image, erosion=const.EROSION,
                     pts_per_region=const.SAMPLES_PER_CELL) -> pandas.DataFrame:
    seeds = {"id": [],
             "pt_y": [],
             "pt_x": []}

    for region in regionprops(regions_image):

        distances = distance_transform_edt(region.image)
        valid_positions = distances > erosion
        if np.sum(valid_positions) < const.MIN_LOCS_PER_CELL:
            continue

        ys, xs, _ = np.nonzero(valid_positions)
        inds = np.random.choice(np.arange(len(ys)), size=pts_per_region, replace=False)
        ys, xs = ys[inds], xs[inds]

        seeds["id"].extend([region.label for _ in ys])
        seeds["pt_y"].extend(ys + region.bbox[0])
        seeds["pt_x"].extend(xs + region.bbox[1])

    seeds_df = pd.DataFrame(seeds)
    return seeds_df


def process_seeding(process_image=None, **kwargs):
    return get_random_seeds(process_image)
