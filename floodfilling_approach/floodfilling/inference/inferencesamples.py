from dataclasses import dataclass
import numpy as np


@dataclass
class InferenceSample:
    input: str
    centers: str
    source: str = "0000"
    label: str = None


