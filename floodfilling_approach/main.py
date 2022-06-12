from floodfilling.training import dataloaders
import matplotlib.pyplot as plt
import numpy as np

splitter = dataloaders.Splitter()
trainloader = dataloaders.Dataloader("train", splitter)

for batch in trainloader:
    inputs = batch.first_pass()[1]
    print(inputs.shape)
    plt.imshow(inputs[0])
    plt.show()

    offsets = np.random.randint(-8, 8, (inputs.shape[0], 2))
    inputs_offset = batch.second_pass(offsets)[1]
    print(inputs_offset.shape)
    plt.imshow(inputs_offset[0])
    plt.show()
