from fastai.vision.all import *
from fastbook import *
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#Shows how to find the L1 and L2 norm
def L1L2(digit, mean):
    dist_abs = (digit - mean).abs().mean()
    dist_sqr = ((digit - mean)**2).mean().sqrt()
    return dist_abs, dist_sqr

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
#plt.imshow(mean7, cmap='binary')
#plt.show()

a_3 = stacked_threes[1]
a_7 = stacked_sevens[1]

dist_3_abs, dist_3_sqr = L1L2(a_3, mean3)
dist_7_abs, dist_7_sqr = L1L2(a_3, mean7)

