
# Code adapted from https://github.com/davidcpage/cifar10-fast

from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
from functools import singledispatch

from tqdm.notebook import tqdm


#####################
# utils
#####################

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t

localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)

class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})


#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

#####################
## graph building
#####################

sep='_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)

def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx,val in enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}


#####################
## training utils
#####################

@singledispatch
def cat(*xs):
    raise NotImplementedError

@singledispatch
def to_numpy(x):
    raise NotImplementedError


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self._stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)

def run_batches(model, writer, epoch, batches, training, n_iter, optimizer_step=None, stats=None, logStep = 25, isTest = False):
    stats = stats or StatsLogger(('loss', 'correct'))
    model.train(training)

    if training:
        with tqdm(total=len(batches)*batches.batch_size, desc='Train Epoch #' + str(epoch)) as t:
            for i, batch in enumerate(batches):
                output = model(batch)
                stats.append(output)
                if training:
                    output['loss'].sum().backward()
                    optimizer_step()
                    model.zero_grad()
                t.set_postfix({'avgAcc':'{:05.3f}'.format(stats.mean('correct')), 'avgLoss':'{:05.3f}'.format(stats.mean('loss'))})
                t.update(batches.batch_size)

                if i % logStep == 0:
                    # Compute accuracy and write values to Tensorboard
                    writer.add_scalar('train/loss', stats.mean('loss'), n_iter)
                    writer.add_scalar('train/acc', stats.mean('correct'), n_iter)

                n_iter += batches.batch_size
    else:
        for batch in batches:
                output = model(batch)
                stats.append(output)
        if isTest:
            writer.add_scalar('test/acc', stats.mean('correct'), 0)
            print("Test Accuracy:", stats.mean('correct'))
        else:
            writer.add_scalar('val/acc', stats.mean('correct'), n_iter)
            writer.add_scalar('val/loss', stats.mean('loss'), n_iter)

    return n_iter

# def train_epoch(model, writer, epoch, train_batches, test_batches, optimizer_step, n_iter):
#     return run_batches(model, writer, epoch, train_batches, True, n_iter, optimizer_step)
#     # run_batches(model, epoch, test_batches, False)

def train(model, writer, optimizer, train_batches, validation_batches, epochs):

    n_iter = 0
    for epoch in range(1, epochs+1):
        run_batches(model, writer, epoch, validation_batches, False, n_iter) # validation step
        n_iter = run_batches(model, writer, epoch, train_batches, True, n_iter, optimizer.step) # train step
        # n_iter = train_epoch(model, writer, epoch, train_batches, test_batches, optimizer.step, n_iter)

def test(model, writer, test_batches):
    print("Evaluating test set")
    run_batches(model, writer, -1, test_batches, training=False, n_iter= 0, isTest=True)