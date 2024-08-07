import string

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *
from fastbook import *
from utils import *


class BasicOptim:
    def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None

#Shows how to find the L1 and L2 norm
def L1L2(digit, mean):
    dist_abs = (digit - mean).abs().mean()
    dist_sqr = ((digit - mean)**2).mean().sqrt()
    return dist_abs, dist_sqr

def mnist_distance(a,b):
    return (a-b).abs().mean((-1,-2))

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

#def linear1(xb):
 #   return xb@weights + bias

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()

#def train_epoch(model):
 #   for xb,yb in dl:
  #      calc_grad(xb,yb,model)
 #       opt.step()
#        opt.zero_grad()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

#def validate_epoch(model):
#accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
#    return round(torch.stack(accs).mean().item(), 4)

#def train_model(model, epochs):
#    for i in range(epochs):
 #       train_epoch(model)
#        print(validate_epoch(model), end=' ')
def get_tensor(path):
    digit = path.ls().sorted()
    digit_tensor = [tensor(Image.open(o)) for o in digit]
    return digit_tensor

path = untar_data(URLs.MNIST)

Path.BASE_PATH = path

zero_tensors = get_tensor((path/'training'/'0'))
one_tensors = get_tensor((path/'training'/'1'))
two_tensors = get_tensor((path/'training'/'2'))
three_tensors = get_tensor((path/'training'/'3'))
four_tensors = get_tensor((path/'training'/'4'))
five_tensors = get_tensor((path/'training'/'5'))
six_tensors = get_tensor((path/'training'/'6'))
seven_tensors = get_tensor((path/'training'/'7'))
eight_tensors = get_tensor((path/'training'/'8'))
nine_tensors = get_tensor((path/'training'/'9'))

#Important, knowing how to stack tensors for finding average values of pixels (mean)
#mean0 = stacked_zeros.mean(0)
#plt.imshow(mean0, cmap='binary')
#plt.show()
stacked_zeros = torch.stack(zero_tensors).float()/255
stacked_ones = torch.stack(one_tensors).float()/255
stacked_twos = torch.stack(two_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_fours = torch.stack(four_tensors).float()/255
stacked_fives = torch.stack(five_tensors).float()/255
stacked_sixes = torch.stack(six_tensors).float()/255
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_eights = torch.stack(eight_tensors).float()/255
stacked_nines = torch.stack(nine_tensors).float()/255

valid_0_tens = torch.stack(get_tensor(path/'testing'/'0')).float()/255
valid_1_tens = torch.stack(get_tensor(path/'testing'/'1')).float()/255
valid_2_tens = torch.stack(get_tensor(path/'testing'/'2')).float()/255
valid_3_tens = torch.stack(get_tensor(path/'testing'/'3')).float()/255
valid_4_tens = torch.stack(get_tensor(path/'testing'/'4')).float()/255
valid_5_tens = torch.stack(get_tensor(path/'testing'/'5')).float()/255
valid_6_tens = torch.stack(get_tensor(path/'testing'/'6')).float()/255
valid_7_tens = torch.stack(get_tensor(path/'testing'/'7')).float()/255
valid_8_tens = torch.stack(get_tensor(path/'testing'/'8')).float()/255
valid_9_tens = torch.stack(get_tensor(path/'testing'/'9')).float()/255

def is_one(x):
    return mnist_distance(x,stacked_ones.mean(0)) < mnist_distance(x, stacked_sevens.mean(0))

print(is_one(valid_1_tens).float().mean())

#train_x = torch.cat([stacked_zeros, stacked_ones, stacked_twos, stacked_threes, stacked_fours,
#                     stacked_fives, stacked_sixes, stacked_sevens, stacked_eights, stacked_nines]).view(-1, 28*28)

#train_y = tensor([0]*len(stacked_zeros) + [1]*len(stacked_ones) + [2]*len(stacked_twos)
#                + [3]*len(stacked_threes) + [4]*len(stacked_fours) + [5]*len(stacked_fives)
#                + [6]*len(stacked_sixes) + [7]*len(stacked_sevens) + [8]*len(stacked_eights)
#                 + [9]*len(stacked_nines)).unsqueeze(1)

#dset = list(zip(train_x,train_y))

#valid_x = torch.cat([valid_0_tens, valid_1_tens, valid_2_tens, valid_3_tens, valid_4_tens, valid_5_tens,
#                     valid_6_tens, valid_7_tens, valid_8_tens, valid_9_tens]).view(-1, 28*28)

#valid_y = tensor([0]*len(valid_0_tens) + [1]*len(valid_1_tens) + [2]*len(valid_2_tens)
#                 + [3]*len(valid_3_tens) + [4]*len(valid_4_tens) + [5]*len(valid_5_tens)
#                 + [6]*len(valid_6_tens) + [7]*len(valid_7_tens) + [8]*len(valid_8_tens)
#                 + [9]*len(valid_9_tens)).unsqueeze(1)

#valid_dset = list(zip(valid_x,valid_y))

#linear_model = nn.Linear(28*28,1)

#weights, bias = linear_model.parameters()

#dl = DataLoader(dset, batch_size=256)
#xb,yb = first(dl)
#valid_dl = DataLoader(valid_dset, batch_size=256)

#dls = DataLoaders(dl, valid_dl)
#lr = 1.

#opt = SGD(linear_model.parameters(), lr)

#simple_net = nn.Sequential(
#    nn.Linear(28*28,30),
#    nn.ReLU(),
#    nn.Linear(30,1)
#)
#learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
#learn.fit(40, 0.1)

#plt.plot(L(learn.recorder.values).itemgot(2))
#plt.show()
