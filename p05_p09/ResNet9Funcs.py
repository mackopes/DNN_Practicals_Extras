
# Code adapted from https://github.com/davidcpage/cifar10-fast

from .core import *
from .torch_backend import *

class QuantizeSymmetric(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None):

        output = input.clone()

        qmin = -1.0 * (2**num_bits)/2
        qmax = -qmin - 1

        if max_value is None or min_value is None:
            max_value = torch.max(input)
            min_value = torch.min(input)

        # compute qparams --> scale and zero_point
        max_val, min_val = float(max_value), float(min_value)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)

        # Computing scale
        if max_val == min_val:
            scale = 1.0
        else:
            max_range = max(-min_val, max_val) # largest mag(value)
            scale = max_range / ((qmax - qmin) / 2)
            scale = max(scale, 1e-8) # for stability purposes

        # Zero_point
        zero_point = 0.0

        # Quantization
        output.div_(scale).add_(zero_point)
        output.round_().clamp_(qmin, qmax)  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None

class Quant(nn.Module):

    def __init__(self, num_bits=8, momentum=0.01):
        super(Quant, self).__init__()
        self.min_val = self.max_val = None
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:

            min_val = self.min_val
            max_val = self.max_val

            if min_val is None or max_val is None:
                # First step executing quantization
                min_val = input.detach().min()
                max_val = input.detach().max()
            else:
                # equivalent to --> min_val = min_val(1-self.momentum) + self.momentum * torch.min(input)
                min_val = min_val + self.momentum * (input.detach().min()  - min_val)
                max_val = max_val + self.momentum * (input.detach().max()  - max_val)

            self.min_val = min_val
            self.max_val = max_val

        return QuantizeSymmetric().apply(input, self.num_bits, self.min_val, self.max_val)

class Conv2dQuant(nn.Conv2d):
    def __init__(self, inCh, outCh, kDim, stride, bias, padding, bits: int = 8):
        super(Conv2dQuant,self).__init__(inCh, outCh, kDim, stride=stride, bias = bias, padding = padding)

        self.QuantInput = Quant(num_bits=bits)
        self.QuantWeights = Quant(num_bits=bits)
        self.QuantOutput = Quant(num_bits=bits)

    def forward(self, input):

        qinput = self.QuantInput(input)
        qweight = self.QuantWeights(self.weight)

        output = nn.functional.conv2d(input, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.QuantOutput(output)


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }

def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),

        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n

def getCIFAR10(batch_size, validation=0.1, doAugment=False):
    DATA_DIR = './datasets'
    dataset = cifar10(root=DATA_DIR, valid_size=validation)

    # Note we've set pad to zero
    train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4 if doAugment else 0))), dataset['train']['labels']))
    val_set = list(zip(transpose(normalise(dataset['val']['data'])), dataset['val']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))

    # Set of data augmentation setps to apply to our dataset.
    # random 32,32 crop from an 32x32 input will give us the image directly
    transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)] if doAugment else [Crop(32,32)]

    train_batches = Batches(Transform(train_set, transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    val_batches = Batches(val_set, batch_size, shuffle=False, drop_last=False)
    test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)

    return train_batches, val_batches, test_batches

def getModel():

    losses = {
        'loss':  (nn.CrossEntropyLoss(reduce=False), [('classifier',), ('target',)]),
        'correct': (Correct(), [('classifier',), ('target',)]),
        }

    model = Network(union(net(), losses)).to(device).half()
    model.to(device)

    return model