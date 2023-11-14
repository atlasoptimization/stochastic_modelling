""" 
The goal of this script is to perform simulation and inference using pyro.
For this, do the following:
    1. Definitions and imports
    2. Build stochastic model
    3. Inference
    4. Plots and illustrations
"""



"""
    1. Definitions and imports ------------------------------------------------
"""


# ii) Imports

import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

# %matplotlib inline
# plt.style.use('default')

# ii) Definitions

n=100
n_simu=5
t=np.linspace(0,1,n)

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.2')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)



"""
    2. Build stochastic model -------------------------------------------------
"""


# i) Simulate some data using true model

mu_true=1
sigma_true=0.1
data=torch.tensor(np.random.normal(mu_true,sigma_true,[n_simu]))

# ii) Invoke observations, latent variables, parameters, model

def model(data=None):
    theta=pyro.param("theta",lambda:torch.randn(()))
    
    with pyro.plate("data",n_simu):
        return pyro.sample("obs",dist.Normal(theta,sigma_true),obs=data)

# pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)



"""
    3. Inference --------------------------------------------------------------
"""


# i) Create the guide

auto_guide = pyro.infer.autoguide.AutoNormal(model)


# ii) Run the optimization

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step(data)
    losses.append(loss)
    if step % 100 == 0:
        logging.info("Elbo loss: {}".format(loss))

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

    
# iii) Sample from distribution

with pyro.plate("samples", 500, dim=-1):
    samples=auto_guide()



"""
    4. Plots and illustrations -----------------------------------------------
"""


# i) Plot the loss

plt.figure(figsize=(5, 2), dpi=300)
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss");






















