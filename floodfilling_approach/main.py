from floodfilling.training import dataloaders
import matplotlib.pyplot as plt

splitter = dataloaders.Splitter()
trainloader = dataloaders.Dataloader("train", splitter)

for batch in trainloader:
    inputs = batch.first_pass()[1]
    print(inputs.shape)
    plt.imshow(inputs[0])

plt.show()
