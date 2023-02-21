import os
import random
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils import train_model, test_model
from load_dataset import load_cifar10_un_list
from model import Net_cifar10


def random_un(classes):
    un_num = []
    random_c = random.sample(range(10), classes)
    un_c = sorted(random_c)
    for c in range(classes):
        un_num.append(random.randint(1, 5000))
    return un_c, un_num


def main(un_c, un_num, classes, ex_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = "../data"
    train, test = load_cifar10_un_list(path, un_c, un_num)
    train_batch_size = 64
    test_batch_size = 64
    train_loader = DataLoader(train, batch_size=train_batch_size, num_workers=4)
    test_loader = DataLoader(test, batch_size=test_batch_size, num_workers=4)
    model = Net_cifar10().to(device)
    lr = 1.0
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        test_model(model, device, test_loader)
        scheduler.step()
    if os.path.exists(f"./{classes}"):
        pass
    else:
        os.makedirs(f"./{classes}")
    torch.save(model.state_dict(), f"./{classes}/cifar10_cnn_{ex_num}.pth")


if __name__ == '__main__':
    ex_total = 50
    for cl in range(10):
        for ex in range(ex_total):
            c_item, num_item = random_un(cl + 1)
            with open(f"config_{cl}.txt", "a+") as f:
                print(f"c:{c_item}", file=f)
                print(f"num:{num_item}", file=f)
            main(c_item, num_item, cl + 1, ex)
