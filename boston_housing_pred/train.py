import numpy as np
import pandas as pd
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from tinygrad.helpers import Timing

from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self):
        self.l1 = nn.Linear(13, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)

    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x

def make_dataset():
    dat = pd.read_csv("boston-housing-dataset.csv")

    X, Y = np.asarray(dat.iloc[:, 1:-1].values).astype(np.float32), np.asarray(dat.iloc[:, -1]).astype(np.float32)
    Y = np.reshape(Y, (Y.shape[0], 1))

    # normalize
    X = (X - X.mean()) / X.std()

    return train_test_split(X, Y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    model = LinearRegression()
    opt = Adam(nn.state.get_parameters(model), lr=0.01)
    X_train, X_test, Y_train, Y_test = make_dataset()

    epochs = 2000
    BS = 8 

    with Tensor.train():
        for step in range(epochs):
            losses = []
            samp = np.random.randint(0, X_train.shape[0], size=(BS))
            batch = Tensor(X_train[samp], dtype=dtypes.float32)
            labels = Tensor(Y_train[samp], dtype=dtypes.float32)

            pred = model(batch)
            loss = pred.sub(labels).square().mean().sqrt()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # calculate accuracy
            acc = (pred == labels).mean()

            if step % 100 == 0:
                print(f"Epoch {step+1}, loss: {loss.numpy()}, acc: {acc.numpy()}")

    with Timing("Time: "):
        predictions = []
        true_labels = []
        avg_acc = 0
        for step in range(100):
            sample = np.random.randint(0, X_test.shape[0], size=(BS))
            batch = Tensor(X_test[sample], requires_grad=False, dtype=dtypes.float32)
            labels = Tensor(Y_test[sample], dtype=dtypes.float32)

            pred = model(batch)

            # calculate accuracy
            avg_acc += (pred.numpy() == labels.numpy()).mean()

            predictions.extend(pred.numpy())
            true_labels.extend(labels.numpy())

        print(f"Test Accuracy: {avg_acc / 100}")

        plt.scatter(predictions, true_labels)
        plt.plot([0, 50], [0, 50], '--k')
        plt.show()

