import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def getPrettyTensorShape(tensor):
    """ if you do "print tensor.shape", you'll get something like "torch.Size([2, 3])" which is not nice.
    By doing "print list(tensor.shape)", you'll get something like "[2L, 3L]" which is still not good.
    In order to print the shape of a tensor nicely, we can do what this function does
    """

    return [elem for elem in np.array(tensor.size())] # this retunrs "[2,3]"

def getArchAsTable(model):

    variablesList = []
    totalSize = 0
    table = PrettyTable(['VarName', 'Shape', 'Size(kB)', 'Size(%)'])
    for name, param in model.named_parameters():
        size = (np.product(list(map(int, param.shape))))/(1024.0) # get variable size in kiloBytes
        variablesList.append([name, getPrettyTensorShape(param), np.round(size, decimals = 2)])
        totalSize += size

    for elem in variablesList:
        table.add_row([elem[0], elem[1], elem[2], np.round(100 * elem[2]/totalSize, decimals=1)])

    return table, totalSize

def showArchAsTable(model, hasName=False):
    """ This function gets as input a model/arch and prints a (pretty) table where each row
    contains one of the trainable variables of the model. The order is kept. In this way, the first row
    represents the entry point of the network.
    The table contains the following fields: variable name, variable shape, variable size (in KB) and ratio to total
    """
    table,totalSize = getArchAsTable(model)
    print("")
    if hasName:
        print("TRAINABLE VARIABLES INFORMATION - arch: %s " % model.name)
    else:
        print("TRAINABLE VARIABLES INFORMATION")
    print(table) # print table
    print("Total (trainable) size: %f kB" % totalSize)
    print("")


def getAccuracy(outputs, labels, num):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return correct.item()/float(num)

class RunningAverage():
    """A simple class that maintains the running average of a quantity """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def appendDateAndTimeToString(string):
    now = datetime.utcnow().strftime("%m_%d_%H_%M_%S")
    return string + "_" + now

def createTensorBoardWriter(resultsDirectory):
    dir = appendDateAndTimeToString(resultsDirectory + '/')
    writer = SummaryWriter(dir)
    print("TensorBoard writer created directory in %s" % dir)
    return writer

def displayImages(trainLoader, n=5):
    idx = np.random.randint(0, len(trainLoader.dataset.indices), n)

    rows = int(np.ceil(n/5))
    fig, axs = plt.subplots(rows, int(n/rows) , figsize=(10, 5*(rows-1)))
    for i in range(len(idx)):
        img = trainLoader.dataset.dataset.data[idx[i]]
        dim = len(img.shape)
        label = trainLoader.dataset.dataset.targets[idx[i]]
        fig.axes[i].imshow(img)
        if dim == 3: # i.e. if it's an image from CIFAR-10 dataset
            fig.axes[i].set_title("Label:" + str(label))
        else:
            fig.axes[i].set_title("Label:" + str(label.item()))

    plt.pause(0.1)