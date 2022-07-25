import json
import time
from dataclasses import dataclass, asdict


# @dataclass
# class SampleInput:
#     image: str = None  # filepath of IF image
#     rna: str = None  # filepath of RNA csv
#
#
# @dataclass
# class SampleLabel:
#     segmentation: str = None  # filepath of cell segmentation
#
#
# @dataclass
# class SampleProperties:
#     density: float = 0  # percent of labeled pixels (any cell) in local region


@dataclass
class Sample:
    id: int  # unique id of example
    source: str  # name of source dataset e.g. 0603
    input: dict  # contains filepaths to input
    label: dict  # contains filepaths to labels
    properties: dict  # contains extra calculated properties
    datetime_generated: str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # time of generation
    split: str = None  # train, val, test, or None


def write_json_samples(path, samples):
    json_string = json.dumps([asdict(samples[i]) for i in samples], indent=2)
    with open(path, "w") as f:
        f.write(json_string)


def load_json_samples(path):
    with open(path, "r") as f:
        json_string = json.loads(f.read())

    examples = {obj['id']: Sample(
        obj['id'],
        obj['source'],
        dict(obj['input']),
        dict(obj['label']),
        dict(obj['properties']),
        obj['datetime_generated'],
        obj['split']
    ) for obj in json_string}

    return examples