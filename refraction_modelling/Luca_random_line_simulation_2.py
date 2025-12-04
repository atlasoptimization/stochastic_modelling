#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to investigate inference and correlations for a situation
similar to the line integration encountered in refraction. A mean of multiple points
along a line is measured; consequently the posteriors for the individual datapoints 
are correlated. For this, do the following:
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
import math
import pyro.distributions as dist
import matplotlib.pyplot as plt


# ii) Definitions

n_pts = 10      # number points
n_ts = 20       # number timeseries

mu_dT_true = torch.linspace(0, 10, n_ts)
sigma_dT_true = torch.linspace(1, 2, n_ts)
sigma_obs = torch.tensor(0.01)

pyro.set_rng_seed(0)


"""
    2. Simulate some data
"""

# i) Draw dT per ts

dT_dist = pyro.distributions.Normal(mu_dT_true, sigma_dT_true)
dT = dT_dist.sample([n_pts]).T


# iii) Draw observation as mean

mean_vals = torch.mean(dT, 1)
obs_dist = pyro.distributions.Normal(loc = mean_vals, scale = sigma_obs)
obs_vals = obs_dist.sample()


"""
    3. Define model
"""


# i) Set up model

ts_plate = pyro.plate('ts_plate', size = n_ts, dim = -2)
pts_plate = pyro.plate('pts_plate', size = n_pts, dim = -1)


# ii) Write model

# def model(obs = None):
#     # Prior
#     mu_dT = pyro.param('mu_dT', init_tensor = torch.ones([n_ts,1]))
#     sigma_dT = pyro.param('sigma_dT', init_tensor = torch.ones([n_ts,1]),
#                          constraint = pyro.distributions.constraints.positive)
    
#     extension_tensor = torch.ones([1, n_pts])

#     dT_dist = pyro.distributions.Normal(loc = mu_dT*extension_tensor,
#                                         scale = sigma_dT*extension_tensor)
#     with ts_plate:    
#         with pts_plate:    
#             dT = pyro.sample('dT', dT_dist)
    
#     # Likelihood
#     mu_obs = torch.mean(dT,1).reshape([n_ts,1])
#     obs_dist = pyro.distributions.Normal(loc = mu_obs, scale = sigma_obs).to_event(1)
#     with ts_plate:    
#         obs = pyro.sample('obs', obs_dist, obs = obs)
        
#     return obs


ts_plate = pyro.plate("ts_plate", n_ts, dim=-1)  # single plate for time

def model(obs=None):
    # Global params per time
    mu_dT    = pyro.param("mu_dT",    torch.ones(n_ts))
    sigma_dT = pyro.param("sigma_dT", torch.ones(n_ts), 
                          constraint = pyro.distributions.constraints.positive)

    # broadcast to [n_ts, n_pts]
    loc   = mu_dT[..., None].expand(n_ts, n_pts)
    scale = sigma_dT[..., None].expand(n_ts, n_pts)

    with ts_plate:
        # points dimension is EVENT (size n_pts)
        dT = pyro.sample("dT", dist.Normal(loc, scale).to_event(1))  # event_dim=1
        mu_obs = dT.mean(dim=-1)                                     # [n_ts]
        obs = pyro.sample("obs", dist.Normal(mu_obs, sigma_obs), obs=obs)  # scalar per time
    return obs


# iii) Pre -train model output

model_output_pretrain = copy.copy(model().detach())



"""
    4. Define guide
"""

# i) Write guide

# def guide(obs=None):
    
#     # a mean value for each of the n_ts * n_ts dT values + a global n_pts x n_pts cov mat
#     mu_dT_post = pyro.param('mu_dT_post', init_tensor = torch.ones([n_ts, n_pts]))
#     cov_dT_post = pyro.param('cov_dT_post', init_tensor = torch.eye(n_pts))

#     dT_post_dist = pyro.distributions.MultivariateNormal(loc = mu_dT_post,
#                                                          covariance_matrix= cov_dT_post)
#     with ts_plate:
#         dT_post_sample = pyro.sample('dT', dT_post_dist)
        
#     return dT_post_sample

def guide(obs=None):
    mu_dT_post = pyro.param("mu_dT_post", torch.ones(n_ts, n_pts))
    cov_dT_post = pyro.param('cov_dT_post', init_tensor = torch.eye(n_pts),
                              constraint = pyro.distributions.constraints.positive_definite)
    dT_post_dist = pyro.distributions.MultivariateNormal(loc=mu_dT_post, covariance_matrix = cov_dT_post)
    with ts_plate:
        dT_post_sample = pyro.sample("dT", dT_post_dist)  # event_dim=1 in both model and guide
    return dT_post_sample

guide_output_pretrain = copy.copy(guide().detach())

# def guide(obs=None):
#     mu_dT_post = pyro.param("mu_dT_post", torch.ones(n_ts, n_pts))
#     L_tril = pyro.param("cov_dT_post_tril",
#                         torch.eye(n_pts),
#                         constraint=pyro.distributions.constraints.lower_cholesky)
#     dT_post_dist = dist.MultivariateNormal(loc=mu_dT_post, scale_tril=L_tril)
#     with ts_plate:
#         dT_post_sample = pyro.sample("dT", dT_post_dist)
        
#     return dT_post_sample


"""
    5. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = obs_vals
loss_sequence = []
for step in range(5000):
    loss = svi.step(data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')


# iii) Post -train model output

model_output_posttrain = copy.copy(model().detach())
guide_output_posttrain = copy.copy(guide().detach())



# iv) Print out the param store

for name, value in pyro.get_param_store().items():
    print('Param : {}; Value : {}'.format(name, value))
    
# post_cov_mat = pyro.get_param_store()['cov_dT_post'].detach()
  

    


"""
    6. Plots and illustrations
"""

# Posterior covariance matrix

# plt.imshow(post_cov_mat)


@torch.no_grad()
def get_guide_params():
    store = pyro.get_param_store()
    mu = store["mu_dT_post"].detach().clone()  # [n_ts, n_pts]
    if "cov_dT_post_tril" in store:
        L = store["cov_dT_post_tril"].detach().clone()  # [n_pts, n_pts] or [n_ts, n_pts, n_pts]
        if L.ndim == 2:
            Cov = L @ L.T                        # [n_pts, n_pts], shared
            Cov = Cov.unsqueeze(0).expand(n_ts, -1, -1).contiguous()
        else:
            Cov = torch.matmul(L, L.transpose(-1, -2))   # [n_ts, n_pts, n_pts]
    else:
        # Fallback for covariance_matrix param name (less stable)
        Cov = store["cov_dT_post"].detach().clone()
        if Cov.ndim == 2:
            Cov = Cov.unsqueeze(0).expand(n_ts, -1, -1).contiguous()
    return mu, Cov  # shapes [n_ts, n_pts], [n_ts, n_pts, n_pts]

@torch.no_grad()
def theoretical_cov(sigma_vec, sigma_obs, n_pts):
    # sigma_vec: [n_pts] prior std per point at a given time
    s2 = sigma_vec**2
    denom = sigma_obs**2 + (s2.sum() / (n_pts**2))
    S = torch.diag(s2)
    # rank-1 correction
    u = s2  # since H^T = (1/n) * 1, the term reduces to (S 1)(1^T S) / denom = (s2 * 1)(1^T * s2)/denom
    # Build full matrix: S - (1/n^2) * (s2 s2^T) / denom
    # (the 1/n^2 is already absorbed into denom above via H S H^T)
    return S - torch.ger(s2, s2) / denom  # [n_pts, n_pts]

# ===== pre/post snapshots =====
mu_pre  = pyro.get_param_store()["mu_dT_post"].detach().clone()
# run training (already done)
mu_post, Cov_post = get_guide_params()

# ===== 1) Covariance heatmaps =====
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(Cov_post[0].cpu());  axs[0].set_title("Guide Cov (t=0)")
axs[1].imshow(Cov_post[n_ts//2].cpu()); axs[1].set_title(f"Guide Cov (t={n_ts//2})")
axs[2].imshow(Cov_post[-1].cpu()); axs[2].set_title("Guide Cov (t=end)")
for ax in axs: ax.set_xlabel("point j"); ax.set_ylabel("point i")
plt.tight_layout()

# ===== 2) Means pre vs post vs truth (pick three times) =====
times = [0, n_ts//2, n_ts-1]
fig, axs = plt.subplots(1, len(times), figsize=(5*len(times), 4), sharey=True)
for k, t in enumerate(times):
    ax = axs[k]
    # ground truth from your simulation 'dT' (shape [n_ts, n_pts])
    ax.plot(dT[t].cpu(), "o", label="true dT", alpha=0.7)
    # pre/post means
    ax.plot(mu_pre[t].cpu(), "x--", label="guide mean (pre)")
    ax.plot(mu_post[t].cpu(), "s-",  label="guide mean (post)", alpha=0.9)
    # 2σ error bars from posterior marginal variances
    std = Cov_post[t].diag().sqrt().cpu()
    ax.fill_between(range(n_pts),
                    (mu_post[t]-2*std).cpu(),
                    (mu_post[t]+2*std).cpu(),
                    alpha=0.2, label="post ±2σ")
    ax.set_title(f"time t={t}")
    ax.set_xlabel("point index")
axs[0].set_ylabel("dT")
axs[-1].legend()
plt.tight_layout()

# ===== 3) Compare learned vs theoretical covariance =====
fig, axs = plt.subplots(1, len(times), figsize=(5*len(times), 4))
for k, t in enumerate(times):
    # prior std per point at time t (from model params)
    sigma_prior_t = pyro.get_param_store()["sigma_dT"].squeeze()[t].repeat(n_pts)
    Cov_th = theoretical_cov(sigma_prior_t, sigma_obs, n_pts)
    im = axs[k].imshow((Cov_post[t]-Cov_th).cpu())
    axs[k].set_title(f"Guide Cov - Theory (t={t})")
    axs[k].set_xlabel("j"); axs[k].set_ylabel("i")
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
plt.tight_layout()




