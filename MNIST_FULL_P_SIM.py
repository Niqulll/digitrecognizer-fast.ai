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

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
def get_tensor(path):
    digit = path.ls().sorted()
    digit_tensor = [tensor(Image.open(o)) for o in digit]
    return digit_tensor

def digit_recognizer(image_tensor):
    digits = []
    for img in image_tensor:
        distances = {}
        for k in means.keys():
            #Same calculation as mnist_distance which is the L1 norm (mean absolute difference)
            distances[k] = (img-means[k]).abs().mean((-1,-2))
        sorted_list = sorted(list(distances.items()), key=lambda x: x[1])
        digits.append(sorted_list[0][0])
    return digits

path = untar_data(URLs.MNIST)

Path.BASE_PATH = path

train_paths = (path/'training').ls().sorted()
valid_paths = (path/'testing').ls().sorted()

train_tensors = {}
for x in train_paths:
    train_tensors[x.name] = [tensor(Image.open(o)) for o in x.ls()]

stacked_tensors = {}
for y in train_tensors.keys():
    stacked_tensors[y] = torch.stack(train_tensors[y]).float()/255

train = {}
valid = {}

for key in stacked_tensors.keys():
    length = len(stacked_tensors[key])
    train_size = math.floor(0.8 * length)
    valid_size = length - train_size
    [train_split, valid_split] = torch.utils.data.random_split(stacked_tensors[key], [train_size, valid_size])
    train_data = [stacked_tensors[key][i] for i in train_split.indices]
    valid_data = [stacked_tensors[key][i] for i in valid_split.indices]
    train_tensor = torch.stack(train_data)
    valid_tensor = torch.stack(valid_data)
    train[key] = train_tensor
    valid[key] = valid_tensor

means = {}

for x in train.keys():
    means[x] = train[x].mean(0)

preds = {}
acc = {}

for k in means.keys():
    preds[k] = digit_recognizer(valid[k])
    accuracy = sum([1 if pred == k else 0 for pred in preds[k]]) / len(preds[k])
    acc[k]= accuracy

total = 0
for digit, accuracy in acc.items():
    total += accuracy
    print(digit, round(accuracy, 3))

print(total/10)
#plt.imshow(means['9'], cmap='binary')
#plt.show()



