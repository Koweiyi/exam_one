import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils import train_model, get_grad
from load_dataset import load_cifar10_single
from model import Net_cifar10


def origin_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"../data"
    g = []
    for i in range(10):
        train = load_cifar10_single(path, i)
        train_batch_size = 1000
        train_loader = DataLoader(train, batch_size=train_batch_size)
        model = Net_cifar10()
        model_path = f"./cifar10_cnn.pth"
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        lr = 1.0
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        epochs = 1
        w0 = copy.deepcopy(model.state_dict()["module.6.weight"])
        W = [w0]
        for epoch in range(1, epochs + 1):
            train_model(model, device, train_loader, optimizer, epoch)
            w_temp = copy.deepcopy(model.state_dict()["module.6.weight"])
            W.append(w_temp)
        grad = get_grad(W)
        g.append(grad)
    with open("grad_origin.txt", "a+") as f:
        print(f"origin_grad:{g}", file=f)


def unlearning_grad(classes, ex_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"../data"
    g = []
    for i in range(10):
        train = load_cifar10_single(path, i)
        train_batch_size = 1000
        train_loader = DataLoader(train, batch_size=train_batch_size)
        model = Net_cifar10()
        model_path = f"./{classes}/cifar10_cnn_{ex_num}.pth"
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        lr = 1.0
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        epochs = 1
        w0 = copy.deepcopy(model.state_dict()["module.6.weight"])
        W = [w0]
        for epoch in range(1, epochs + 1):
            train_model(model, device, train_loader, optimizer, epoch)
            w_temp = copy.deepcopy(model.state_dict()["module.6.weight"])
            W.append(w_temp)
        grad = get_grad(W)
        g.append(grad)
    with open("grad_performance.txt", "a+") as f:
        print(f"unlearning_grad:{g}", file=f)


if __name__ == '__main__':
    ex_total = 50
    for cl in range(10):
        with open("grad_performance.txt", "a+") as fl:
            print(f"unlearning_classes:{cl + 1}", file=fl)
        for ex in range(ex_total):
            unlearning_grad(cl + 1, ex)
