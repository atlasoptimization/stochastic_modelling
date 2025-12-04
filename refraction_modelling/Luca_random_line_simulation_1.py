#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to simulate datapoints using a "set of lines" model
and infer the parameters of this model (mean, spread of set of lines) from these
datapoints with pyro. For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Define model
    4. Define guide
    5. Perform inference
    6. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import copy
import matplotlib.pyplot as plt


# ii) Definitions

n_obs = 100
d_range = [0,100]

mu_a_true = torch.tensor(1.0)
sigma_a_true = torch.tensor(0.3)
sigma_obs = torch.tensor(0.01)

pyro.set_rng_seed(0)


"""
    2. Simulate some data
"""

# i) Draw distances

d_dist = pyro.distributions.Uniform(low = d_range[0], high = d_range[1])
d_vals = d_dist.sample([n_obs])


# ii) Draw from set of lines

a_dist = pyro.distributions.Normal(loc = mu_a_true, scale = sigma_a_true)
a_vals = a_dist.sample([n_obs])


# iii) Draw observation

beta_vals = a_vals * d_vals
obs_dist = pyro.distributions.Normal(loc = beta_vals, scale = sigma_obs)
obs_vals = obs_dist.sample()


"""
    3. Define model
"""


# i) Set up model

obs_plate = pyro.plate('obs_plate', size = n_obs, dim = -1)


# ii) Write model

def model(d_input, obs = None):
    # Prior
    mu_a = pyro.param('mu_a', init_tensor = torch.tensor([0.0]))
    sigma_a = pyro.param('sigma_a', init_tensor = torch.tensor([1.0]),
                         constraint = pyro.distributions.constraints.positive)
    a_dist = pyro.distributions.Normal(loc = mu_a, scale = sigma_a)
    
    with obs_plate:
        a = pyro.sample('a_sample', a_dist)
    
    # Likelihood
    obs_dist = pyro.distributions.Normal(loc = a*d_input, scale = sigma_obs)
    with obs_plate:
        obs = pyro.sample('obs', obs_dist, obs = obs)
        
    return obs
    

# iii) Pre -train model output

model_output_pretrain = copy.copy(model(d_vals).detach())



"""
    4. Define guide
"""

# i) Write guide

# def guide(d_input, obs = None):
#     pass

# guide = pyro.infer.autoguide.AutoNormal(model)

def guide(d_input, obs=None):

    # Variational parameters for q(a_sample) = Normal(loc, scale)
    mu_a_post = pyro.param("mu_a_post", torch.zeros(n_obs))
    sigma_a_post = pyro.param("sigma_a_post", torch.ones(n_obs), 
                              constraint = pyro.distributions.constraints.positive)
    post_dist =  pyro.distributions.Normal(mu_a_post, sigma_a_post)
    
    with obs_plate:
        a_sample = pyro.sample("a_sample", post_dist)
        
    return a_sample



"""
    5. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (d_vals, obs_vals)
loss_sequence = []
for step in range(1500):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')


# iii) Post -train model output

model_output_posttrain = copy.copy(model(d_vals).detach())



# iv) Print out the param store

for name, value in pyro.get_param_store().items():
    print('Param : {}; Value : {}'.format(name, value))
    
    
# v) Sample completely new dataset
@torch.no_grad()
def sample_new_dataset(M=100, d_range=(0., 100.)):
    # 1) new inputs
    d_new = pyro.distributions.Uniform(d_range[0], d_range[1]).sample((M,))

    # 2) read learned hyperparameters
    mu_hat = pyro.get_param_store()["mu_a"].squeeze()      # shape []
    sig_hat = pyro.get_param_store()["sigma_a"].squeeze()  # shape []

    # 3) sample new slopes and observations
    a_new = pyro.distributions.Normal(mu_hat, sig_hat).sample((M,))
    y_new = pyro.distributions.Normal(a_new * d_new, sigma_obs).sample()

    return d_new, y_new

# usage after training:
d_sim, y_sim = sample_new_dataset(M=200, d_range=(0., 100.))


"""
    6. Plots and illustrations
"""


# i) Scatterplot of data and model outputs

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)

# i) Scatterplot of observed data
axes[0].scatter(d_vals, obs_vals, alpha=0.7)
axes[0].set_title("Observed Data")
axes[0].set_xlabel("d values")
axes[0].set_ylabel("Observed beta")

# ii) Model output before training
axes[1].scatter(d_vals, model_output_pretrain, alpha=0.7)
axes[1].set_title("Model Output (Pre-Training)")
axes[1].set_xlabel("d values")
axes[1].set_ylabel("Model output beta")

# iii) Model output after training
axes[2].scatter(d_vals, model_output_posttrain, alpha=0.7)
axes[2].set_title("Model Output (Post-Training)")
axes[2].set_xlabel("d values")
axes[2].set_ylabel("Model output beta")

# iii) Model output after training
axes[3].scatter(d_sim, y_sim, alpha=0.7)
axes[3].set_title("Model Output resimulated")
axes[3].set_xlabel("d values")
axes[3].set_ylabel("Model output beta")

plt.tight_layout()
plt.show()

