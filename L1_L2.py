import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *
from fastbook import *


#Shows how to find the L1 and L2 norm
def L1L2(digit, mean):
    dist_abs = (digit - mean).abs().mean()
    dist_sqr = ((digit - mean)**2).mean().sqrt()
    return dist_abs, dist_sqr

def mnist_distance(a,b):
    return (a-b).abs().mean((-1,-2))


path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]

#Important, knowing how to stack tensors for finding average values of pixels (mean)
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

#Important, how to get the mean of stacked tensors
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

#Important how to actually display the info with matplotlib
plt.imshow(mean7, cmap='binary')
plt.show()

a_3 = stacked_threes[1]
a_7 = stacked_sevens[1]

dist_3_abs, dist_3_sqr = L1L2(a_3, mean3)
dist_7_abs, dist_7_sqr = L1L2(a_3, mean7)

def is_3(x):
    return mnist_distance(x,mean3) < mnist_distance(x,mean7)

valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255

valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255

accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

print(accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2)