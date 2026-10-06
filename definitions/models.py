import torch
from torch import nn



class MLP(nn.Module):
    # Expects the input.shape = (B, 3, 28, 28)
    def __init__(self, num_h1, num_h2, num_classes):
        super().__init__()
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*28*28,num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, num_classes)
        )
    def forward(self,x):
        return self.classifier(x)



def conv_layer(ch_in, ch_out, conv_kernel, pool_kernel):

    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=(conv_kernel,conv_kernel)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(pool_kernel,pool_kernel))
    )


class CNN(nn.Module):
    def __init__(self, ch_in, ch1, ch2, conv_ker_1, pool_ker_1, conv_ker_2, pool_ker_2,num_hid,num_classes):
        super().__init__()

        size = 28

        size = size - conv_ker_1 + 1
        size = size // pool_ker_1

        size = size - conv_ker_2 + 1
        size = size // pool_ker_2

        num_flat = ch2 * size * size

        self.feature_extractor=nn.Sequential(
            conv_layer(ch_in, ch1, conv_ker_1, pool_ker_1),
            conv_layer(ch1, ch2, conv_ker_2, pool_ker_2),
            nn.Flatten()        
        )
        self.classifier=nn.Sequential(
            nn.Linear(num_flat, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid,num_classes)
        )

        # self.box=nn.Sequential(
        #     conv_layer(ch_in, ch1, conv_ker_1, pool_ker_1),
        #     conv_layer(ch1, ch2, conv_ker_2, pool_ker_2),
        #     nn.Flatten(),        
        #     nn.Linear(num_flat, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid,num_classes)
        # )



    def forward(self, x):
        x=self.feature_extractor(x)
        x=self.classifier(x)
        # x=self.box(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self,num_in):
        super().__init__()
        self.classifier=nn.Linear(num_in,1)
    def forward(self, x):
        return self.classifier(x)