from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d, Linear

class LeNet:
    def __init__(self):
        # l1 = conv layer with 6 feature maps of 28x28 connected to 5x5 neighbourhood in the input
        # l2 = sub sampling layer with 6 feature maps of size 14x14 connected to a 2x2 neighbourhood of feature map l1
        # l3 = conv layer with 16 feature maps 
        # l4 = sub sampling layer w 16 feature maps of size 5x5
        # l5 = conv layer with 120 feature maps w size 1x1 (actually linear layer for this input size)
        # l6 = fully connected to C5 with 84 units
        self.conv1 = Conv2d(1, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16*5*5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)
    
    def __call__(self, x):
        x = self.conv1(x).relu()
        x = x.avg_pool2d(kernel_size=(2, 2))
        x = self.conv2(x).relu()
        x = x.avg_pool2d(kernel_size=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).log_softmax(axis=1)
        return x
