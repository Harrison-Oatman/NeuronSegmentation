TRAINING_DIR = "C:\\Lab Work\\segmentation\\floodfilling_data\\examples\\"
INFERENCE_DIR = "C:\\Lab Work\\segmentation\\floodfilling_data\\inference\\"

WINDOW_SIZE = 137
DELTA_MAX = 12

TRAIN_VAL_SPLIT = 0.8

BATCH_SIZE_TRAIN = 28
BATCH_SIZE_VAL = 16  # not in use
INPUT_ATTR = "default"
INPUT_CHANNELS = 3

NET_PATH = "C:\\Lab Work\\segmentation\\saved_models\\"
CURRENT_NET_PATH = "C:\\Lab Work\\segmentation\\saved_models\\most_recent\\"
LOG_PATH = "logs/fit/"

MIN_PROCESSES = 2
EROSION = 1
SAMPLES_PER_CELL = 1
MIN_LOCS_PER_CELL = 10

VISIT_THRESHOLD = 0.7
TRUE_LABEL_VALUE = 0.95
FALSE_LABEL_VALUE = 0.05
