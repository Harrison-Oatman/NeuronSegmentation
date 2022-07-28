from ..const import *
import os
import json
from .inferenceloader import InferenceLoader
from ..utils.cropping import imalign
import numpy as np
import time


def unpack_json(json_path, overwrite):
    if os.path.exists(json_path) and not overwrite:
        with open(json_path) as f:
            examples = json.loads(f.read())

    else:
        examples = {}

    return examples


def slice_to_tuple(slices: list[slice]):
    ans = []
    for s in slices:
        ans.append([s.start, s.stop, s.step])
    return ans


class InferenceLogger:

    def __init__(self, loader: InferenceLoader, output_path=INFERENCE_OUTPUT_PATH,
                 overwrite=OVERWRITE_INFERENCE_OUTPUT):

        self.json_path = output_path + "data.json"
        self.source = loader.sample.source
        self.base_image = loader.image
        self.json_log = unpack_json(self.json_path, overwrite)
        self.examples = []
        self.examples_path = output_path + "examples\\"
        self.id = 0

        if not os.path.exists(output_path + "examples\\"):
            os.mkdir(output_path + "examples\\")

    def write_pom(self, pom, center):
        # a_slice, b_slice = imalign(self.base_image, pom, center)

        pom_path = f"{self.examples_path}pom_{self.source}_{self.id}.npy"

        example = dict()
        example['inference'] = pom_path
        example['centery'] = str(center[0])
        example['centerx'] = str(center[1])
        example['date_created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        example['id'] = self.id

        np.save(pom_path, pom)

        self.examples.append(example)
        self.save()

        self.id += 1

    def save(self):
        self.json_log[self.source] = self.examples
        jsons = json.dumps(self.json_log)
        with open(self.json_path, "w") as f:
            f.write(jsons)

