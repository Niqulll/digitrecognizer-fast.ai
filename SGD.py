import torch
from fastai.vision.all import *
from fastbook import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def pr_eight(x,w):
    return (x*w).sum()

def f(x):
    return x**2

plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red')
plt.show()