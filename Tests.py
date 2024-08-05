import string

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import *
from fastbook import *
from utils import *

def test_func(a,b):
    lst = list(zip(a,b))
    return lst

x = [1,2,3,4]
y = 'abcd'

print(test_func(x,y))