import numpy as np


def crop(data, center, crop_shape):
    expanded_data = np.expand_dims(data, [0])

    cropped = batch_crop(expanded_data, center, crop_shape)

    if cropped is not None:
        return cropped[0]
    return None


def get_offset_slice(center, crop_shape, data):
    start = center - crop_shape // 2
    end = start + crop_shape

    in_range = np.all(np.bitwise_and((start >= 0), (end <= data.shape[1:-1])), axis=-1)
    if not in_range:
        return None

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector + [slice(None)])  # skip the final dimension
    return selector


def batch_crop(data, center, crop_shape):
    """
    crops an array from the data to the specified paste_shape
    at the specified center, intended for single samples
    """

    center = np.array(center)
    crop_shape = np.array(crop_shape)

    # calculate start and end
    selector = get_offset_slice(center, crop_shape, data)

    if selector is None:
        return None

    cropped = data[selector]

    return cropped


def crop_offset(data, offsets, crop_shape):
    centers = np.array(data.shape[1:-1], dtype=np.int32) // 2
    centers += offsets
    return batch_crop(data, centers, crop_shape)


def batch_crop_offset(data_list, offset_arr, crop_shape):
    inputs = []
    for i in range(len(data_list)):
        inputs.append(crop_offset(np.array(data_list[i:i + 1]),
                                  offset_arr[i],
                                  crop_shape))
    return inputs


def batch_paste(data, center, paste):
    """
    crops an array from the data to the specified paste_shape
    at the specified center, intended for single samples
    """

    center = np.array(center)
    paste_shape = np.array(paste.shape)

    # calculate start and end
    selector = get_offset_slice(center, paste_shape[1:-1], data)
    data[selector] = paste

    return data


def paste_offset(data, offsets, paste):
    centers = np.array(data.shape[1:-1]) // 2
    centers += offsets
    return batch_paste(data, centers, paste)


def batch_paste_offset(old_data, offset, new_patch):
    inputs = []
    for i in range(len(old_data)):
        inputs.append(paste_offset(np.array(old_data[i:i + 1]),
                                  offset[i],
                                  new_patch[i:i+1]))
    return inputs
