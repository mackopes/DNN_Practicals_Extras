import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
import time
import numpy as np

from tqdm.notebook import tqdm

from . import miscFuncs as misc

# Normalising the images
def normaliseData():
#     print("Preprocessing: Normalise data")
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def cifar10Trans(doAugment=False):
#     print("Preprocessing: Crop/Mirror/Normalise data")
    if doAugment:
        return  transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        return  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

def getMNISTdataset(batchSize=128, numWorkers=4):

    # directory where the MNIST lives (it will be downloaded here if not found)
    pathToDatasets = './datasets/'

    # Dataset objects
    trainset = torchvision.datasets.MNIST(root=pathToDatasets, train=True, download=True, transform=normaliseData())
    testset = torchvision.datasets.MNIST(root=pathToDatasets, train=False, download=True, transform=normaliseData())

    # Dataset Loaders
    trainloader = data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    testloader = data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)

    return trainloader, testloader


def getCIFAR10dataset(pathToDatasets = './datasets/', batchSize=128, valid_size=0.1, augment=True):

    pathToData = pathToDatasets

    train_dataset = torchvision.datasets.CIFAR10(root=pathToData, train=True, download=True, transform=cifar10Trans(doAugment=augment))
    test_dataset = torchvision.datasets.CIFAR10(root=pathToData, train=False, download=True, transform=cifar10Trans())

    num_train = len(train_dataset)
    split = int(np.floor(valid_size * num_train))

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[num_train - split,split])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, device, train_loader, writer, optimiser, epoch, n_iter, vizStep=100, logStep=25):
    model.train() # this enables parameter updates during backpropagation as well
                  # as other updates such as those in batch-normalisation layers

    loss_avg = misc.RunningAverage()
    acc_avg = misc.RunningAverage()
    # for every mini-batch containing batch_size images...
    with tqdm(total=len(train_loader.dataset), desc='Train Epoch #' + str(epoch)) as t:
        for i , (data, target) in enumerate(train_loader):

            inputs, labels = data.to(device), target.to(device)
            # send the data (images, labels) to the device (either CPU or GPU)

            # zero gradients from previous step
            optimiser.zero_grad()

            # this executes the forward() method in the model
            outputs = model(inputs)

            # compute loss
            loss = model.criterion(outputs, labels)

            # backward pass
            loss.backward()

            # evaluate trainable parameters
            optimiser.step()

            # Monitoring progress, accuracy and loss
            acc_avg.update(misc.getAccuracy(outputs, labels, inputs.shape[0]))
            loss_avg.update(loss.item())
            t.set_postfix({'avgAcc':'{:05.3f}'.format(acc_avg()), 'avgLoss':'{:05.3f}'.format(loss_avg())})
            t.update(data.shape[0])

            if i % logStep == 0:
                # Compute accuracy and write values to Tensorboard
                acc = misc.getAccuracy(outputs, labels, inputs.shape[0])
                writer.add_scalar('train/loss', loss.item(), n_iter)
                writer.add_scalar('train/acc', acc, n_iter)

            n_iter += inputs.shape[0]

    return n_iter

def test(model, device, test_loader, writer):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            # count total images in batch
            total += labels.size(0)
            # count number of correct images
            correct += (predicted == labels).sum()

    test_acc = correct.item()/float(total)
    writer.add_scalar('test/acc', test_acc, 0)

    print("Accuracy on Test Set: %.4f" % test_acc)

def evalValidation(model, device, valLoader, writer, n_iter):

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = []

        for data in valLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            total_loss.append(model.criterion(outputs, labels).item())

            _, predicted = torch.max(outputs.data, 1)
            # count total images in batch
            total += labels.size(0)
            # count number of correct images
            correct += (predicted == labels).sum()

    acc = correct.item()/float(total)

    avg_loss = sum(total_loss)/float(len(total_loss))

    # adding accuracy and loss to TensorBoard
    writer.add_scalar('val/acc', acc, n_iter)
    writer.add_scalar('val/loss', avg_loss, n_iter)

    return acc, avg_loss


def defaultInit(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out') # using He, K. et al. (2015) initialization
            torch.nn.init.xavier_normal_(m.weight)
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    print("Model initalised successfully")
