import torch
from fastai.vision.all import *
from fastbook import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def pr_eight(x,w):
    return (x*w).sum()

def f(t, params):
    a,b,c = params
    return a*(t**2)+(b*t)+c

def mse(preds, targets):
    return ((preds-targets)**2).mean()

time = torch.arange(0,20).float();
speed = torch.randn(20)*3+0.75*(time-9.5)**2+1

def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

params = torch.randn(3).requires_grad_()
orig_params = params.clone()

lr = 2e-5

def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds

for i in range(10):
    apply_step(params)

params = orig_params.detach().requires_grad_()
_,axs = plt.subplots(1,4,figsize=(12,3))

for ax in axs:
    show_preds(apply_step(params, False), ax)
plt.tight_layout()
plt.show()