import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import art.attacks.evasion as evasion
from art.estimators.classification import PyTorchClassifier
from typing import Tuple, Optional
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import backbone.resnet as resnet
from train import Model
import argparse

print(torch.cuda.is_available())


class Test_Dataset(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._len = len(x)

    def __getitem__(self, item): 
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

 

def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_set, y_set = [], []
    for i, (x, y) in enumerate(test_loader):
        x_set.append(x)
        y_set.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_set_tensor = torch.cat(x_set)
    y_set_tensor = torch.cat(y_set)

    if n_examples is not None:
        x_set_tensor = x_set_tensor[:n_examples]
        y_set_tensor = y_set_tensor[:n_examples]

    return x_set_tensor, y_set_tensor, torch.min(x_set_tensor), torch.max(x_set_tensor)



criterion = nn.CrossEntropyLoss()
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def load_cifar10(
    n_examples: Optional[int] = None,
    trainset: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    dataset = torchvision.datasets.CIFAR10(root='Datasets/cifar10',
                               train=trainset,
                               transform=transform,
                               download=False)
    return _load_dataset(dataset, n_examples)


def accuracy(model, images, targets, device):
    total = 0
    correct = 0
    test_loader = DataLoader(Test_Dataset(images.float(), targets.float()),
                             batch_size=10, shuffle=False, drop_last=True, num_workers=0)
    with torch.no_grad():
        for data in test_loader:
            imgs, label = data[0].to(device),data[1].to(device)
            outputs = model(imgs.float())
            _, predicted = torch.max(outputs.data,1) # max value and index
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc_test = float(100*correct/total)
    return acc_test


def run_fgsm_experiment(models,epsilon,colors):
    color_idx = 0
    for m in models:
        print(m)
        # load pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model.load_from_checkpoint(m).to(device).eval()
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # get test set
        x_test, y_test, min_pixel_value, max_pixel_value = load_cifar10()
        print(x_test.size())
        # x_test = np.array(x_test)
        # Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )
        x_test = x_test.numpy()
        # print(type(x_test))
        acc_eps = []
        for i in range(len(epsilon)):
            atk = evasion.FastGradientMethod(estimator=classifier, eps=epsilon[i]) 
            x_test_adv = atk.generate(x=x_test)
            x_test_adv_torch = torch.from_numpy(x_test_adv)
            print(x_test_adv_torch.size())
            acc = accuracy(model, x_test_adv_torch, y_test, device)
            acc_eps.append(acc)
            if i == 0:
                print("Accuracy on benign test examples: {}%".format(acc))
            else:
                print("Accuracy on FSGM adversarial test examples: {}%".format(acc), '\n eps: ', i  , '/255')
        plt.plot(epsilon, acc_eps, color=colors[color_idx], label=m)
        color_idx = color_idx + 1
        plt.xlabel("eps")
        plt.ylabel("Accuracy")
        plt.legend()
    plt.show()
    plt.savefig('FGSM_acc.png')




iteration = [1,2,3,4,5,6,7,8,9,10]
def run_pgd_experiment(models,epsilon,colors):
    color_idx = 0
    for m in models:
        print(m)
        # load pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model.load_from_checkpoint(m).to(device).eval()
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # get test set
        x_test, y_test, min_pixel_value, max_pixel_value = load_cifar10()
        x_test = np.array(x_test)
        # Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            # optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )
        acc_eps = []
        for i in range(len(epsilon)):
            print(2.5*epsilon[i]/10.0)
            atk = evasion.ProjectedGradientDescent(batch_size=100, num_random_init=0,
                                           estimator=classifier, eps=epsilon[i],
                                           eps_step=2.5*epsilon[i]/10.0, max_iter=10)
            x_test_adv = atk.generate(x=x_test)
            x_test_adv_torch = torch.from_numpy(x_test_adv)
            acc = accuracy(model, x_test_adv_torch, y_test, device)
            acc_eps.append(acc)
            if i == 0:
                print("Accuracy on benign test examples: {}%".format(acc))
            else:
                print("Accuracy on PGD adversarial test examples: {}%".format(acc), '  [ eps: ', i, ' ]' )
        plt.plot(iteration, acc_eps, color=colors[color_idx], label=m)
        color_idx = color_idx + 1
        plt.xlabel("epsilon")
        plt.ylabel("Accuracy")
        plt.title("PGD (step size = 2/255, max_iteration = 10)")
        plt.legend()
    plt.show()
    plt.savefig('PGD_acc.png')



def run_pgdl2_experiment(models,epsilon,colors):
    for m in models:
        print(m)
        # load pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model.load_from_checkpoint(m).to(device).eval()
        criterion = nn.CrossEntropyLoss()
        # get test set
        x_test, y_test, min_pixel_value, max_pixel_value = load_cifar10()
        x_test = np.array(x_test)
        # Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )
        acc_eps = []
        atk = evasion.ProjectedGradientDescent(batch_size=100,norm =2, num_random_init=0,
                                       estimator=classifier, eps=128/255,
                                       eps_step=0.05, max_iter = 100)
        x_test_adv = atk.generate(x=x_test)
        x_test_adv_torch = torch.from_numpy(x_test_adv)
        acc = accuracy(model, x_test_adv_torch, y_test, device)
        acc_eps.append(acc)
        print("Accuracy on PGD_L2 adversarial test examples: {}%".format(acc))



def main(args):

    print("train classifier")
    epsilon = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255]
    models = [args.ckpt]
    colors = ['blue', 'green', 'orange', 'yellow']

    run_fgsm_experiment(models,epsilon,colors)
    run_pgd_experiment(models,epsilon,colors)    
    # run_pgdl2_experiment(models,epsilon,colors)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--ckpt', type=str,
                    help='checkpoint of a model')

   
    args = parser.parse_args()

    main(args)