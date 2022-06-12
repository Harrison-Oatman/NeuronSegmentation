from floodfilling.training import dataloaders
import matplotlib.pyplot as plt
import numpy as np
from floodfilling.model.model import ConvStack2DFFN
import tensorflow as tf

splitter = dataloaders.Splitter()
train_loader = dataloaders.Dataloader("train", splitter)

model = ConvStack2DFFN(input_shape=(67, 67, 3))
model.compile(optimizer="adam", loss=tf.nn.sigmoid_cross_entropy_with_logits,
              metrics=[])

for i in range(4):
    for batch in train_loader:
        inputs, labels = batch.first_pass()
        labels = np.array(labels, dtype=np.float32)
        inputs = np.array(inputs, dtype=np.float32)
        inputs /= 255.
        return_dict = model.train_on_batch(inputs, labels, return_dict=True)
        print(return_dict)

fig, axes = plt.subplots(3, 3)
for i in range(3):
    axes[i, 0].imshow(inputs[i])
    axes[i, 1].imshow(model(inputs[i:i+1])[0])
    axes[i, 2].imshow(labels[i])
plt.show()
