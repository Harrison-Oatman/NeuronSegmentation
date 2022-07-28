import matplotlib.pyplot as plt
import numpy as np
from ..const import *
from ..sampling.training_input import *
from ..sampling.training_labels import *
from ..sampling.sampling import expand_dir, load_data_from_src
from .traininginputs import *
import os
import json
import time
from tqdm import tqdm


def main():
    json_path = INFERENCE_OUTPUT_PATH + "processed.json"
    with open(json_path) as f:
        datasets = json.loads(f.read())

    make_samples_from_dataset(datasets)


def load_example_data(example):
    example_data = dict()
    example_data["pom"] = np.load(example["inference"])
    example_data["branch_im"] = np.load(example["branch_im"])
    example_data["branch_seg"] = np.load(example["branch_seg"])
    return example_data


def make_samples_from_dataset(datasets, data_dir=DATA_DIR, trn_dir=BRANCH_SAMPLE_PATH, overwrite=False,
                              # seeding_fn=process_seeding,
                              input_fn=simple_inputs, input_name="default",
                              label_fn=branch_labels, label_name="default", ):
    # TODO: allow for examples to have multiple labels, inputs?

    # start json samples list
    samples = {}
    json_path = trn_dir + "samples.json"

    if os.path.exists(json_path):
        if not overwrite:
            with open(json_path) as f:
                samples = json.loads(f.read())

    count = 0

    for source in datasets.keys():

        if source in [sample["source"] for sample in samples.values()]:
            print("dataset already parsed, returning...")
            continue

        examples = datasets[source]

        # load data to be used by functions
        data = load_data_from_src(data_dir+source+"//")

        # generate input, labels
        input_data = input_fn(**data)
        label_data = label_fn(**data)

        for example in tqdm(examples):

            pairs = example["pairs"]

            if len(pairs) == 0:
                continue

            example_data = load_example_data(example)
            cropped_input = example_crop(example, example_data, input_data)
            cropped_label = example_crop(example, example_data, label_data)

            for pair in pairs:
                input_frame = pair_crop(pair, cropped_input)
                label_frame = pair_crop(pair, cropped_label)
                pom_frame = pair_crop(pair, example_data["pom"])
                branch_im_frame = pair_crop(pair, example_data["branch_im"])
                branch_seg_frame = pair_crop(pair, example_data["branch_seg"])

                # determine if humans have labeled these regions the same
                label = pair_label(pair, label_frame, branch_im_frame)
                if label is None:
                    continue

                output = separate_data(pair, pom_frame, branch_seg_frame)

                # make folders for each image type
                expand_dir(trn_dir, [str(count)])
                d = f"{trn_dir}{str(count)}\\"

                input_path = d + "input.npy"
                poms_path = d + "pom.npy"

                sample = dict()
                sample["source"] = source
                sample["datetime created"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sample["input"] = input_path
                sample["poms"] = poms_path
                sample["label"] = 1 if label else 0
                sample["id"] = count

                np.save(input_path, input_frame)
                np.save(poms_path, output)

                samples[count] = sample

                count += 1

    with open(json_path, "w") as f:
        f.write(json.dumps(samples))
