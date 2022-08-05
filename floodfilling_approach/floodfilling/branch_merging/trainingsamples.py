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

    make_samples_from_dataset(datasets, overwrite=True)


def load_branch_json_samples(path=BRANCH_SAMPLE_PATH+"samples.json"):
    with open(path) as f:
        return json.loads(f.read())


def write_branch_json_samples(samples, json_path=BRANCH_SAMPLE_PATH+"samples.json"):
    with open(json_path, "w") as f:
        f.write(json.dumps(samples, indent=4))


def load_example_data(example):
    example_data = dict()
    example_data["pom"] = np.load(example["inference"])
    example_data["threshold"] = np.load(example["threshold"])
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
    rows = np.zeros((1, N_RNA_SPECIES))

    if os.path.exists(json_path):
        if not overwrite:
            samples = load_branch_json_samples(json_path)

    count = 0

    for source in datasets.keys():

        # if source in [sample["source"] for sample in samples.values()]:
        #     print("dataset already parsed, returning...")
        #     continue

        examples = datasets[source]

        # load data to be used by functions
        data = load_data_from_src(data_dir+source+"//")

        # generate input, labels
        input_data = input_fn(**data)
        label_data = label_fn(**data)

        for example in tqdm(examples):

            pairs = example["pairs"]
            example_viewed = True

            if len(pairs) == 0:
                continue

            example_data = load_example_data(example)
            cropped_input = example_crop(example, example_data, input_data)
            cropped_label = example_crop(example, example_data, label_data)
            cropped_rna = rna_crop(example, example_data, data)
            rna = rna_vecs(cropped_rna, example_data)

            for pair in pairs:
                # crop data to frame
                label_frame = pair_crop(pair, cropped_label)
                branch_im_frame = pair_crop(pair, example_data["branch_im"])

                # determine if humans have labeled these regions the same
                label = pair_label(pair, label_frame, branch_im_frame)
                if label is None:
                    example_viewed = True
                    continue

                # crop remaining data tp frame
                pom_frame = pair_crop(pair, example_data["pom"])
                branch_seg_frame = pair_crop(pair, example_data["branch_seg"])
                input_frame = pair_crop(pair, cropped_input)

                output = separate_data(pair, pom_frame, branch_seg_frame)

                # make folders for example data
                expand_dir(trn_dir, [str(count)])
                d = f"{trn_dir}{str(count)}\\"

                input_path = d + "input.npy"
                poms_path = d + "pom.npy"
                rna_path = d + "rna.npy"

                sample = dict()
                sample["source"] = source
                sample["datetime created"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sample["input"] = input_path
                sample["poms"] = poms_path
                sample["label"] = 1 if label else 0
                sample["id"] = count
                sample["pair"] = pair
                sample["rna_vecs"] = rna_path

                np.save(input_path, input_frame)
                np.save(poms_path, output)
                np.save(rna_path, rna)

                samples[count] = sample

                count += 1

            if example_viewed:
                rows = np.vstack((rows, rna))

    np.save(PCA_SAVE_PATH+"rows.npy", rows)
    write_branch_json_samples(samples, json_path)
