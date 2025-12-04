#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to generate noisy data that is centered around a parabola
and export this data into an excel file.
For this, do the following:
    1. Imports and definitions
    2. Generate Data
    3. Export to excel
"""


"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ii) Definitions

n_data = 100
x = torch.linspace(0,1,n_data)

alpha_1 = 1
alpha_2 = 1
alpha_3 = -1

"""
    2. Generate data
"""


# i) Trend

mu = alpha_1 + alpha_2 * x + alpha_3 * x**2


# ii) Noise 

obs_dist = pyro.distributions.Normal(mu, 0.1)
with pyro.plate('obs_plate', dim = -1):
    obs = pyro.sample('obs', obs_dist)

obs = obs.detach()


"""
    3. Export to excel
"""


# i) illustrate

plt.plot(obs)


# ii) Set up dirs

obs_df  = pd.DataFrame({'x' : x, 'y': obs})
obs_df.to_excel('quadratic_data.xlsx', sheet_name='sheet1', index=False)
obs_df.to_csv('quadratic_data.csv')

