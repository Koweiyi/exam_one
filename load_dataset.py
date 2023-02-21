import random
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import transforms


def load_cifar10(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = random.sample(range(n), n)
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar10_un_list(path, un_c: list, un_num: list):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = set(range(n))
    for i, j in zip(un_c, un_num):
        un = random.sample(range(5000 * i, 5000 * i + 5000), j)
        n_train = n_train - set(un)
    n_train = list(n_train)
    n_train = random.sample(n_train, len(n_train))
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar10_single(path, c):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m_test = random.sample(range(1000 * c, 1000 * c + 1000), 1000)
    test_dataset = Subset(test_dataset, m_test)
    return test_dataset


if __name__ == '__main__':
    pass
