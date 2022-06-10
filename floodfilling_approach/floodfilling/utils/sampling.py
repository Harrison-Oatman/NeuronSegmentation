import json
from dataclasses import dataclass, asdict
import time
import numpy as np
import cv2
import pandas as pd
import os
from .. import const
from .masking import crop


@dataclass
class SampleInput:
    image: str = None  # filepath of IF image
    rna: str = None  # filepath of RNA csv


@dataclass
class SampleLabel:
    segmentation: str = None  # filepath of cell segmentation


@dataclass
class SampleProperties:
    density: float = 0  # percent of labeled pixels (any cell) in local region


@dataclass
class Sample:
    id: int  # unique id of example
    source: str  # name of source dataset e.g. 0603
    input: SampleInput  # contains filepaths to input
    label: SampleLabel  # contains filepaths to labels
    properties: SampleProperties  # contains extra calculated properties
    datetime_generated: int = time.time()  # time of generation
    split: str = None  # train, val, test, or None


def write_json_examples(path, examples):
    json_string = json.dumps([asdict(examples[i]) for i in examples], indent=2)
    with open(path, "w") as f:
        f.write(json_string)


def load_json_examples(path):
    with open(path, "r") as f:
        json_string = json.loads(f.read())

    examples = {obj['id']: Sample(
        obj['id'],
        obj['source'],
        SampleInput(obj['input']['image'], obj['input']['rna']),
        SampleLabel(obj['label']['segmentation']),
        SampleProperties(obj['properties']['density']),
        obj['datetime_generated'],
        obj['split']
    ) for obj in json_string}

    return examples


def make_image_example(image, centroid):
    slice_size = const.WINDOW_SIZE + 2*const.DELTA_MAX
    return crop(image, centroid, (slice_size, slice_size))


def make_if_example(if_image, centroid, trn_dir, example_id):
    filepath = f'{trn_dir}if_images\\{example_id}_if_image.npy'
    cropped_image = make_image_example(if_image, centroid)

    if cropped_image is not None:
        np.save(filepath, cropped_image)
        return filepath
    return None


def make_label(cell_image, centroid, trn_dir, example_id, example_cell_id):
    filepath = f'{trn_dir}segmentations\\{example_id}_segmentation.npy'
    cropped_image = make_image_example(cell_image, centroid)

    if cropped_image is not None:
        segmentation = cropped_image == example_cell_id
        np.save(filepath, segmentation)
        return filepath, cropped_image

    return None


def make_examples_from_dataset(src_dir, trn_dir=const.TRAINING_DIR, overwrite=False):
    cell_image = np.expand_dims(np.load(src_dir + "cell_image.npy"), -1)
    if_image = np.array(cv2.imread(src_dir + "Map2TauImage.png"))
    centroids = pd.read_csv(src_dir + "centroids.csv")

    examples = {}
    json_path = trn_dir + "examples.json"

    source = src_dir[-4:]

    if os.path.exists(json_path):
        if not overwrite:
            examples = load_json_examples(json_path)
            if source in [example.source for example in examples.values()]:
                print("dataset already parsed, returning...")
                return

    if not os.path.isdir(trn_dir + "if_images"):
        os.mkdir(trn_dir + "if_images")

    if not os.path.isdir(trn_dir + "segmentations"):
        os.mkdir(trn_dir + "segmentations")

    for i, point in centroids.iterrows():
        new_id = 0
        if len(examples) > 0:
            new_id = max(example_id for example_id in examples) + 1

        centroid = (int(np.floor(point.soma_centroid_y)), int(np.floor(point.soma_centroid_x)))

        if_path = make_if_example(if_image, centroid, trn_dir, new_id)
        if if_path is None:
            continue

        label_path, cropped_label = make_label(cell_image, centroid,
                                               trn_dir, new_id, int(point.cell_id))

        density = np.average(cropped_label >= 0)

        input_ = SampleInput(if_path)
        label = SampleLabel(label_path)
        properties = SampleProperties(density)

        example = Sample(new_id, source, input_, label, properties)

        examples[new_id] = example

    write_json_examples(json_path, examples)


def main():
    import matplotlib.pyplot as plt
    # example_a = Sample(0, "src", SampleInput(), SampleLabel(), SampleProperties())
    #
    # write_json_examples("C:\\Lab Work\\segmentation\\floodfilling_data\\0520\\file.json", [example_a])
    # examples = load_json_examples("C:\\Lab Work\\segmentation\\floodfilling_data\\0520\\file.json")

    make_examples_from_dataset("C:\\Lab Work\\segmentation\\floodfilling_data\\0520\\")
    # print(const.WINDOW_SIZE)

    examples = load_json_examples(const.TRAINING_DIR+"examples.json")
    print(type(examples[0].split))
    plt.imshow(np.load(examples[0].input.image))
    plt.show()


if __name__ == '__main__':
    main()

