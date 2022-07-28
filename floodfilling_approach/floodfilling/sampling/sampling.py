import numpy as np
import cv2
import pandas as pd
import os
from .. import const
from .samples import Sample, write_json_samples, load_json_samples
from ..utils.cropping import crop
from .training_input import simple_inputs
from .training_labels import branch_labels
from .seeds import process_seeding
from ..inference import inferencesamples

import json
from dataclasses import asdict


def example_crop(image, centroid):
    slice_size = const.WINDOW_SIZE + 2 * const.DELTA_MAX
    return crop(image, centroid, (slice_size, slice_size))


def get_center(image):
    y, x, _ = image.shape
    return image[y//2, x//2, 0]


def make_example_input(if_image, centroid, trn_dir, example_id, cat_dir="inputs"):
    filepath = f'{trn_dir}{cat_dir}\\{example_id}.npy'
    cropped_image = example_crop(if_image, centroid)

    if cropped_image is not None:
        np.save(filepath, cropped_image)
        return filepath
    return None


def make_example_label(label_data, centroid, trn_dir, example_id, example_cell_id=None):
    filepath = f'{trn_dir}labels\\{example_id}_segmentation.npy'
    cropped_image = example_crop(label_data, centroid)

    if cropped_image is not None:
        segmentation = cropped_image == get_center(cropped_image)
        np.save(filepath, segmentation)
        return filepath, cropped_image

    return None


def expand_dir(path, names):
    """
    expands the directory specified in 'path' to include
    subdirectories with names given by 'names'
    """
    for name in names:
        if not os.path.isdir(path + name):
            os.mkdir(path + name)


def load_data_from_src(src_dir):
    print("loading data...")
    data = {
        "cell_image": np.expand_dims(np.load(src_dir + "cell_image.npy"), -1),
        "body_image": np.expand_dims(np.load(src_dir + "body_image.npy"), -1),
        "process_image": np.expand_dims(np.load(src_dir + "process_image.npy"), -1),
        "process_names": np.load(src_dir + "process_names.npy"),
        "if_image": np.array(cv2.imread(src_dir + "Map2TauImage.png")),
        "rna": pd.read_csv(src_dir + "barcodes.csv")
    }
    print("loaded")
    return data


def make_samples_from_dataset(src_dir, trn_dir=const.TRAINING_DIR, overwrite=False,
                              seeding_fn=process_seeding,
                              input_fn=simple_inputs, input_name="default",
                              label_fn=branch_labels, label_name="default", ):
    # TODO: allow for examples to have multiple labels, inputs?

    # load data to be used by functions
    data = load_data_from_src(src_dir)

    # start json samples list
    samples = {}
    json_path = trn_dir + "samples.json"

    source = src_dir[-4:]

    if os.path.exists(json_path):
        if not overwrite:
            samples = load_json_samples(json_path)
            if source in [example.source for example in samples.values()]:
                print("dataset already parsed, returning...")
                return

    # make folders for each image type
    expand_dir(trn_dir, ["inputs", "labels"])

    # calculate seeds
    seed_df = seeding_fn(**data)

    # generate input, labels
    input_data = input_fn(**data)
    label_data = label_fn(**data)

    for i, seed in seed_df.iterrows():
        # determine id of example
        new_id = 0
        if len(samples) > 0:
            new_id = max(example_id for example_id in samples) + 1

        # clean location of seed
        location = (int(np.floor(seed.pt_y)), int(np.floor(seed.pt_x)))

        # make example input, if possible
        input_path = make_example_input(input_data, location, trn_dir, new_id)
        if input_path is None:
            continue

        # make example label
        label_path, cropped_label = make_example_label(label_data, location,
                                                       trn_dir, new_id)

        density = np.average(cropped_label > 0)

        # create new sample with input, labels, properties
        input_dict = {input_name: input_path}
        label_dict = {label_name: label_path}
        properties_dict = {"density": density}

        sample = Sample(new_id, source, input_dict, label_dict,
                        properties_dict)

        samples[new_id] = sample

    write_json_samples(json_path, samples)


def make_inference_test_from_dataset(src_dir, test_dir=const.INFERENCE_DIR,
                                     seeding_fn=process_seeding, seed_name="default",
                                     input_fn=simple_inputs, input_name="default",
                                     label_fn=branch_labels, label_name="default"
                                     ):
    # load data to be used by functions
    data = load_data_from_src(src_dir)

    source = src_dir.rstrip("\\")[-4:]

    # calculate seeds
    seed_df = seeding_fn(**data)
    seed_path = test_dir + f"seeds_{source}_{seed_name}.csv"
    seed_df.to_csv(seed_path)

    # generate input, labels
    input_data = input_fn(**data)
    label_data = label_fn(**data)

    # choose filepath for inference data
    input_path = test_dir + f"inputs_{source}_{input_name}.npy"
    label_path = test_dir + f"labels_{source}_{label_name}.npy"

    # save inference data
    np.save(input_path, input_data)
    np.save(label_path, label_data)

    inference_example = inferencesamples.InferenceSample(
        input=input_path,
        centers=seed_path,
        source=source,
        label=label_path
    )

    json_str = json.dumps(asdict(inference_example), indent=2)

    with open(test_dir + source + "_inference.json", "w") as f:
        f.write(json_str)


def main(k, train=True):
    import matplotlib.pyplot as plt
    # example_a = Sample(0, "src", SampleInput(), SampleLabel(), SampleProperties())
    #
    # write_json_samples("C:\\Lab Work\\segmentation\\floodfilling_data\\0520\\file.json", [example_a])
    # samples = load_json_samples("C:\\Lab Work\\segmentation\\floodfilling_data\\0520\\file.json")

    if train:
        make_samples_from_dataset(f"C:\\Lab Work\\segmentation\\floodfilling_data\\{k}\\", overwrite=False)
    # print(const.WINDOW_SIZE)

        examples = load_json_samples(const.TRAINING_DIR + "samples.json")
        for i in range(5, 25):
            plt.imshow(np.load(examples[i].label['default']))
            plt.show()

    else:
        make_inference_test_from_dataset(f"C:\\Lab Work\\segmentation\\floodfilling_data\\inference\\{k}\\")
    # for i in range(5, 25):
    #     plt.imshow(np.load(examples[i].label['default']))
    #     plt.show()


if __name__ == '__main__':
    main()
