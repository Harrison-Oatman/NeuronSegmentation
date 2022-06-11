import numpy as np


def crop(data, center, crop_shape):
    expanded_data = np.expand_dims(data, 0)

    cropped = batch_crop(expanded_data, center, crop_shape)

    if cropped is not None:
        return cropped[0]
    return None


def batch_crop(data, center, crop_shape):
    """
    crops an array from the data to the specified crop_shape
    at the specified center, intended for single samples
    """

    center = np.array(center)
    crop_shape = np.array(crop_shape)

    # calculate start and end
    start = center - crop_shape // 2
    end = start + crop_shape

    in_range = np.all(np.bitwise_and((start >= 0), (end <= data.shape[1:-1])), axis=-1)
    if not in_range:
        return None

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector + [slice(None)])  # skip the final dimension
    cropped = data[selector]

    return cropped


def crop_offset(data, offsets, crop_shape):
    centers = np.array(data.shape[1:-1]) // 2
    centers += offsets
    return batch_crop(data, centers, crop_shape)


