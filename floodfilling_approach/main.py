from floodfilling.sampling import foldercleaning, sampling
from floodfilling.model.resnet import ConvStack2DFFN
from floodfilling.model import ffn
from floodfilling.traincontrol import TrainController
from floodfilling.inferencecontroller import InferenceController
from floodfilling.branch_merging import branches, trainingsamples, branchloader
import tensorflow as tf
from tensorflow import keras
from floodfilling.const import *


def main():
    # branch()
    # trainingsamples.main()
    # branch()
    # inference()
    btl = branchloader.BranchTrainLoader("train", branchloader.BranchSplitter())



def branch():
    branches.main()


def folder_cleaning():

    for src in ["0520", "0602", "0605", "0613", "0807", "0812", "0814",
                "1014", "1028", "1030", "1102", "1107"]:
        foldercleaning.main(src)
        sampling.main(src)

    foldercleaning.main("0225", False)
    sampling.main("0520", True)


def train_from_scratch():

    net = ConvStack2DFFN(input_shape=(67, 67, 3), depth=8, k=5)
    model = ffn.FFN(net)
    model.net = net
    model.compile()
    controller = TrainController(first_step_grad=True)
    controller.train(model, epochs=200, depth=None)


def retrain():

    net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-15_13-42-10\\9"
    net = keras.models.load_model(net_path)

    model = ffn.FFN(net)
    controller = TrainController(first_step_grad=True)
    controller.train(model, epochs=200, depth=None)


def inference():

    # net_path = "C:\\Lab Work\\segmentation\\saved_models\\most_recent"
    # net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-21_18-57-08\\final"
    net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-06_17-29-08\\final"
    # # net_path = "C:\\Lab Work\\segmentation\\saved_models\\2022-07-21_14-49-11\\14"
    net = keras.models.load_model(net_path, custom_objects={
        "sigmoid_cross_entropy_with_logits_v2": tf.nn.sigmoid_cross_entropy_with_logits
    })
    model = ffn.FFN(net, train=False)

    for src in ["0520", "0602", "0605", "0613", "0807", "0812", "0814",
                "1014", "1028", "1030", "1102", "1107"
                ]:
        foldercleaning.main(src, False)
        sampling.main(src, False)

        controller = InferenceController(INFERENCE_DIR + src + "_inference.json")
        controller.inference(model)

    branches.main()


if __name__ == '__main__':
    main()