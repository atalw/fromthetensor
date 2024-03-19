import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn import optim 
from tinygrad.nn.state import get_parameters
from extra.datasets import fetch_cifar
from extra.training import train

# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
class AlexNet():
    def __init__(self):
        # adjusting conv values for 32x32 input size
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def __call__(self, x):
        x = self.conv1(x).relu()
        x = self.bn1(x).max_pool2d(kernel_size=3, stride=2)
        x = self.conv2(x).relu()
        x = self.bn2(x).max_pool2d(kernel_size=3, stride=2)
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()
        x = x.max_pool2d(kernel_size=3, stride=2)
        x = x.reshape(shape=[x.shape[0], -1])
        # x = x.avg_pool2d(output_size=(6,6))
        # x = x.avg_pool2d(kernel_size=(6,6))
        x = self.fc1(x).relu().dropout(0.5)
        x = self.fc2(x).relu().dropout(0.5)
        x = self.fc3(x).relu()
        return x.log_softmax()


X_train, Y_train, X_test, Y_test = fetch_cifar()
X_train = X_train.reshape((-1, 3, 64, 64))
Y_train = Y_train.reshape((-1,))

model = AlexNet()
opt = optim.SGD(get_parameters(model), lr=0.01, momentum=0.9, weight_decay=0.0005)
epochs = 100
BS = 128 

with Tensor.train():
    for step in range(epochs):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        batch = X_train.numpy()[samp]
        labels = Y_train.numpy()[samp]

        out = model(Tensor(batch.astype(np.float32), requires_grad=False))
        loss = Tensor.sparse_categorical_crossentropy(out, Tensor(labels.astype(np.float32)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = out.argmax(axis=1).numpy()
        acc = (pred == labels).mean()
        print(f"Epoch {step+1}, loss: {loss.numpy()}, acc: {acc}")
