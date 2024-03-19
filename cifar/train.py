import numpy as np
from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.nn import optim 
from tinygrad.nn.state import get_parameters
from extra.datasets import fetch_cifar
from extra.training import train
from tqdm import trange
from tinygrad.helpers import CI

# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
class AlexNet():
    def __init__(self):
        # adjusting conv values for 32x32 input size
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=1, padding=1)
        self.fc1 = nn.Linear(256*4*4, 4096)
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
        x = x.dropout(0.5)
        x = self.fc1(x).relu()
        x = x.dropout(0.5)
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        return x.log_softmax(axis=1)

def eval():
    sample = np.random.randint(0, X_test.shape[0], size=(BS))
    batch = Tensor(X_test.numpy()[sample], requires_grad=False, dtype=dtypes.float32)
    labels = Tensor(Y_test.numpy()[sample], requires_grad=False)

    out = model(batch)
    loss = Tensor.sparse_categorical_crossentropy(out, labels)

    correct = out.argmax(axis=1) == labels.argmax()
    
    return correct, loss

def train_step():
    sample = np.random.randint(0, X_train.shape[0], size=(BS))
    batch = Tensor(X_train.numpy()[sample], requires_grad=False, dtype=dtypes.float32)
    labels = Tensor(Y_train.numpy()[sample], requires_grad=False)

    out = model(batch)
    loss = Tensor.sparse_categorical_crossentropy(out, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()


X_train, Y_train, X_test, Y_test = fetch_cifar()
# load data and label into GPU and convert to dtype accordingly
X_train, X_test = X_train.to(device=Device.DEFAULT).float(), X_test.to(device=Device.DEFAULT).float()
Y_train, Y_test = Y_train.to(device=Device.DEFAULT), Y_test.to(device=Device.DEFAULT)
X_train, X_test = X_train.reshape((-1, 3, 32, 32)), X_test.reshape((-1, 3, 32, 32))
# Y_train, Y_test = Y_train.one_hot(10), Y_test.one_hot(10)

model = AlexNet()
epochs = 100
BS = 128 

with Tensor.train():
    lr = 0.01
    for round in range(10):
        opt = optim.SGD(get_parameters(model), lr=lr, momentum=0.9, weight_decay=0.0005)
        if round > 0:
            corrects = []
            losses = []
            for step in range(10):
                correct, loss = eval()
                losses.append(loss.numpy().tolist())
                corrects.extend(correct.numpy().tolist())
            
            correct_sum, correct_len = sum(corrects), len(corrects)
            eval_acc = correct_sum / correct_len * 100
            print(f"Eval        Round: {round} Test Accuracy: {correct_sum}/{correct_len} = {eval_acc:2f}% Loss {(sum(losses)/len(losses)):2f}")

        for i in (t := trange(epochs, disable=CI)):            
            train_step()
        lr /= 10
