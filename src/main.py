import torch
import numpy as np
from data import gen_data
import os
import deepxde as dde
from modelGc import FourierDeepONet as Gc
from modelGp import FourierDeepONet as Gp
# ensure that the device is set to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pretrained neural operators
net_Gc = Gc(num_parameter=1, width=32, modes1=50, modes2=30).to(device)

#change the path to your lacation of saved GC model
save_path_Gc = 'Gc_model.pt'
checkpoint_Gc = torch.load(save_path_Gc, map_location=device)
net_Gc.load_state_dict(checkpoint_Gc["model_state_dict"])
net_Gc.eval()

net_Gp = Gp(num_parameter=1, width=48, modes1=30, modes2=30).to(device)
#change the path to your lacation of saved GP model
save_path_Gp = 'Gp_model.pt'
checkpoint_Gp = torch.load(save_path_Gp, map_location=device)
net_Gp.load_state_dict(checkpoint_Gp["model_state_dict"])
net_Gp.eval()

# clear gradients
dde.gradients.clear()

def dataset(listV):
    X_train = []
    Y_train = []
    for j in listV:
        xx = gen_data(j, 1)[3][:, 60:181].reshape(-1, 1)
        yy = gen_data(j, 1)[4][:, 60:181].reshape(-1, 1)
        vv = np.linspace(j, j, 12221).reshape(-1, 1)
        X = np.append(np.append(xx, yy, axis=1), vv, axis=1).astype(np.float32)
        X_train.append([X])
        Y_train.append(gen_data(0.9, 3000)[0].reshape(-1, 1))
    X_train = np.concatenate(X_train, axis=0).reshape(-1, 3)
    Y_train = np.concatenate(Y_train, axis=0).reshape(-1, 1)
    return [X_train, Y_train]


# dataset
data_list = [0.1, 0.3, 0.5, 0.2, 0.3, 0.4, 0.6, 0.8,0.7, 0.9]  #inlet velocities
X_train, Y_train = dataset(data_list)

# Neural networt parameters
net = dde.nn.FNN([3] + [140] * 2 + [1], activation='relu',kernel_initializer="Glorot uniform")
data = dde.data.dataset.DataSet(X_train, Y_train, X_train, Y_train)
model = dde.Model(data, net)


def main(alpha, lambd, weight, iterations, lr, conv):
    zero = torch.from_numpy(np.array(0)).to(device)
    # Used to train the neural network

    def loss_func(y_true, train_pre):

        YY = train_pre.reshape(-1, 12221).T
        index = YY.shape[1]
        list1 = [j for j in np.arange(index)]
        opt1, opt2, opt3, opt4 = zero, zero, zero, zero
        for i in list1:

            Y_index = YY[:, i:i+1]
            Y_mean = Y_index-torch.mean(Y_index)
            gama = torch.sigmoid(alpha*Y_mean)
            velocity = data_list[i]
            X_trunk = torch.from_numpy(
                np.array([velocity]).astype(np.float32)[None, :]).to(device)
            X_branch = gama.reshape(101, 121)[None, None, :, :]

            y_pred_c = net_Gc((X_branch, X_trunk)).reshape(101, 201)
            y_pred_p = net_Gp((X_branch, X_trunk)).reshape(101, 201)

            # objective of concentration
            covz_pred = torch.mean(abs(y_pred_c-y_pred_c.mean())**2)

            # objective of pressure
            predp_in1 = (torch.mean(y_pred_p[:, 0:1]))
            predp_in2 = (torch.mean(y_pred_p[0:1, 21:40]))
            predp_out = (torch.mean(y_pred_p[:, 200:201]))
            diffp_pred = 0.01*predp_in1+0.002*predp_in2-0.01*predp_out  # area weighted

            # solid volumn
            V_solid =(torch.sum(1-gama)-1200)/12000
            # objectives and inequility constraint
            opt1 = opt1+covz_pred
            opt2 = opt2+weight*diffp_pred
            opt3 = opt3-lambd*min(V_solid, zero)

        opt = opt1+opt2+opt3
        return opt/index  # average value of 9 differnent v

    # used to display each term in the loss function
    def loss_func1(y_true, train_pre):
        opt1, opt2, opt3, opt4 = 0, 0, 0, 0
        YY = torch.from_numpy(train_pre.reshape(-1, 12221).T).to(device)
        index = YY.shape[1]

        for i in [j for j in np.arange(index)]:

            Y_index = YY[:, i:i+1]
            Y_mean = Y_index-torch.mean(Y_index)
            gama = (torch.sigmoid(alpha*Y_mean))
            velocity = data_list[i]
            X_trunk = torch.from_numpy(
                np.array([velocity]).astype(np.float32)[None, :]).to(device)
            X_branch = gama.reshape(101, 121)[None, None, :, :]

            y_pred_c = net_Gc((X_branch, X_trunk)).reshape(101, 201).cpu()
            y_pred_p = net_Gp((X_branch, X_trunk)).reshape(101, 201).cpu()

            predp_in1 = 0.01*(torch.mean(y_pred_p[:, 0:1])).cpu()
            predp_in2 = 0.002*(torch.mean(y_pred_p[0:1, 21:40])).cpu()
            predp_out = 0.01*(torch.mean(y_pred_p[:, 200:201])).cpu()
            diffp_pred = predp_in1+predp_in2-predp_out

            covz_pred = torch.mean(abs(y_pred_c-y_pred_c.mean())**2)

            V_solid = (torch.sum(1-gama)-1200)/12000
            opt1 = opt1+dde.utils.to_numpy(covz_pred)
            opt2 = opt2+dde.utils.to_numpy(diffp_pred)
            opt3 = opt3-dde.utils.to_numpy(min(V_solid, zero))
            opt4 = opt4+dde.utils.to_numpy(abs(V_solid))

        return [opt1/index, opt2/index, opt3/index, opt4/index]

    model.compile('adam', lr=lr, loss=loss_func, decay=("step", 5000, 0.9),
                  metrics=[lambda y_true, y_pred: loss_func1(y_true, y_pred)[0],
                           lambda y_true, y_pred: loss_func1(
                               y_true, y_pred)[1],
                           lambda y_true, y_pred: loss_func1(
                               y_true, y_pred)[2],
                           lambda y_true, y_pred: loss_func1(y_true, y_pred)[3]])
    path = 'model_task'
    if not os.path.exists(path):
        os.makedirs(path)
    checker = dde.callbacks.ModelCheckpoint(f"{path}/model", save_better_only=False, period=1)
    
    losshistory, train_state = model.train(iterations=iterations, display_every=1000, callbacks=[checker])
    model.restore(f"F:/case-pytorch/FNO_Topo/222src/topology/model_task/model-{train_state.best_step}.pt", verbose=1)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    
    alpha=10
    weight = 0.1
    lambd = 1
    lr = 1e-3
    iterations=100000

    main(alpha, lambd, weight, iterations, lr,1e-5) #[0.1, 0.5, 0.9]

