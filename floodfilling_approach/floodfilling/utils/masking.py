import numpy as np


def crop(data, center, crop_shape):
    """
    crops an array from the data to the specified crop_shape
    at the specified center
    """
    center = np.array(center)
    crop_shape = np.array(crop_shape)

    # calculate start and end
    start = center - crop_shape // 2
    end = start + crop_shape

    if (not np.all(start >= 0)) or (not np.all(end <= data.shape[:-1])):
        return None

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple(selector + [slice(None)])  # skip the final dimension
    cropped = data[selector]

    return cropped
