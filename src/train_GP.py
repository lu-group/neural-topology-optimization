import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import torch
import numpy as np
from model_GP import FourierDeepONet
from data_GP import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dataset(dde.data.Data):

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (self.train_x[0][indices], self.train_x[1][indices],), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y

def main():

    X_train,y_train = dataset('train')  
    X_test, y_test = dataset('test')  
    data = Dataset(X_train, y_train, X_test, y_test)

    net = FourierDeepONet(num_parameter=X_train[1].shape[1], width=48, modes1=30, modes2=30, regularization=["l2", 3e-6])
    model = dde.Model(data, net)
    path = 'model_dataset_task'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    def loss_func_L1(y_true, y_pred):
        return torch.nn.L1Loss()(y_pred, y_true)

    def loss_func_L2(y_true, y_pred):
        return torch.nn.MSELoss()(y_pred, y_true)

    model.compile("adam", lr=1e-3, loss=loss_func_L1, decay=("step", 5000, 0.9),
                  metrics=[lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
                           lambda y_true, y_pred: np.sqrt(np.mean(((y_true - y_pred) ** 2)))], )
    checker = dde.callbacks.ModelCheckpoint(f"{path}/model", save_better_only=False, period=1000)
    losshistory, train_state = model.train(iterations=30000, batch_size=32, display_every=1000, callbacks=[checker]) #
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":

    main()
