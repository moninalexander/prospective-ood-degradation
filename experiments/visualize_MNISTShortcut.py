import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from definitions.data import ShortcutMNIST
from math import isclose
from pathlib import Path
from configs.local_config import DATA_ROOT


mnist_train_dataset=MNIST(root=DATA_ROOT,train=True,download=True, transform=ToTensor())
mnist_test_dataset=MNIST(root=DATA_ROOT,train=False,download=True, transform=ToTensor())

prob_correct=1
prob_wrong=0
shortcut_H=1
shortcut_W=1

marked_train_dataset=ShortcutMNIST(
    mnist_train_dataset,
    prob_correct_shortcut=prob_correct,
    prob_wrong_shortcut=prob_wrong,
    shortcut_H=shortcut_H,
    shortcut_W=shortcut_W,
)

marked_test_dataset=ShortcutMNIST(
    mnist_test_dataset,
    prob_correct_shortcut=prob_correct,
    prob_wrong_shortcut=prob_wrong,
    shortcut_H=shortcut_H,
    shortcut_W=shortcut_W,
)

num_rows=5
num_cols=5

indices=torch.randint(len(marked_train_dataset),(num_rows*num_cols,))

print(indices)



fig, axes = plt.subplots(num_rows,num_cols,figsize=(12,12))
for indx, ax in enumerate(axes.flatten()):

    X,y=marked_train_dataset[indices[indx]]
    ax.imshow(X.permute(1,2,0))
    ax.axis('off')

fig.suptitle('Marked data examples',fontsize=20)

plt.show()
