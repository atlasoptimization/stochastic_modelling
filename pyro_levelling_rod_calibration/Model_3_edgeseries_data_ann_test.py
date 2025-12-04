#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test if the tilt_ann can recuperate nonlinear effects
given different net architectures. This amounts to a very reduced levelling rod
calibration with edgeseries data. For this, we import synthetically generated data
and use it to train a probabilistic model that abstra ts away most of the complexities
of model_3.
For this, do the following:
    1. Imports and definitions
    2. Support functions
    3. Build Model
    4. Build guide
    5. Perform inference
    6. Plots and illustration
    
Since we work with synthetic data for which the ground truth alpha is known, we
can evaluate the success of this model by measuring how close we come to the
true parameters and latents.

Written by Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

import string
import pyro
import torch
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2 
import arviz as az
from pathlib import Path


# ii) Definitions

n_tilt_types = 2
n_edge_max = 89
n_rods = 100
n_datapoints = 2*n_rods

x = (1/n_edge_max)*torch.arange(0, n_edge_max)
input_vars = [x for k in range(n_rods)]


"""
    2. Build simple dataset
"""

# observations will be one edgeseries 

# i) Tilt types data

tilt_types_perstaff = [torch.tensor([0.0, 1.0]) for k in range(n_rods)]
input_data = [{'x': x, 'tilt_type': tilt_types_perstaff[k]} for k in range(n_rods)]
tilt_gt_0 = 10*x**2
tilt_gt_1 = 10 +10*x - 10*x**2
tilt_gt = [[tilt_gt_0, tilt_gt_1] for k in range(n_rods)]

tilt_types_perobs = [torch.tensor([0.0]) for k in range(n_rods)] + [torch.tensor([1.0]) for k in range(n_rods)]
tilt_gt_perobs = [tilt_gt_0 for k in range(n_rods)] + [tilt_gt_1 for k in range(n_rods)] 

# observations are just one edgeseries per datapoint
noise_dist = pyro.distributions.Normal(0,1)
observations_perstaff = [[tilt_gt_0 + noise_dist.sample([n_edge_max]), 
                 tilt_gt_1 + noise_dist.sample([n_edge_max])] for k in range(n_rods)] 
observations = ([observations_perstaff[k][0] for k in range(n_rods)] 
                    + [observations_perstaff[k][1] for k in range(n_rods)])


"""
    3. Build Model
"""


# i) Tilt model - ANN

class ANN(torch.nn.Module):
    # Will output a different edge series based on input tilt type.
    def __init__(self):
        # Initialize instance using init method from base class
        super().__init__()
                        
        self.models = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(1, 1),
                torch.nn.Tanh(),
                torch.nn.Linear(1, 1),
                torch.nn.Tanh(),
                torch.nn.Linear(1, 1)
            ) for _ in range(n_tilt_types)
        ])
        
                
    def forward(self, tilt_type, x):
        # Reshape to account for batch shape
        x = x.reshape([n_edge_max, -1])
        
        
        nonlinear_drift = self.models[tilt_type](x)
        
        
        # tilt_types = tilt_types.reshape([-1])

        # Choose model based on tilt type and apply
        # n_obs, n_edge = x.shape
        # nonlinear_drift = torch.zeros([n_obs, n_edge])
        # for tt in range(n_tilt_types):
            # Find observations of this tilt type
            # mask = (tilt_types == tt)
            
            # Shape x inputs and pass
            # x_masked = x[mask,:].reshape([-1,1])      
            # nonlinear_drift[mask,:] = self.models[tt](x_masked).reshape([-1, n_edge_max])
            
        return nonlinear_drift


tilt_ann = ANN()

def add_tilt_effect(tilt_type_list, n_meas, n_meas_max, n_edge_rod_k, n_edge_max):
    # ANN nonlinear effect dependent on tilt type
    tilt_series = torch.zeros(n_meas_max, n_edge_max)
    x = (1/n_edge_rod_k)*torch.arange(0, n_edge_max) 
    # TODO! This should be of dim n_obs, n_edge, not 1, n_edge,... or maybe this is already done correctly
    # but is maybe the dim wrong, gets batched over firs or second dim in fwd?
    # xx = x.repeat([len(tilt_type_list),1])
    
    for k, tilt_type in enumerate(tilt_type_list):
        tilt_series[k,:] = tilt_ann(tilt_type, x)
    
    return tilt_series


def model(input_vars, observations = None):
    # Mark the parameters inside of the ann for optimization
    pyro.module("tilt_ann", tilt_ann)
    
    # General params
    sigma_cal = pyro.param("sigma_cal", init_tensor = 10 * torch.eye(1),
                   constraint = pyro.distributions.constraints.positive)
    
    # Addition of tilt effect
    with pyro.plate('obs_plate', size = n_obs, dim = -1) as k:
        # Set up drifts
        tilt_types_k = tilt_types_perobs[k]    
        nonlinear_drift = add_tilt_effect(tilt_types_k, 2, 2, n_edge_max, n_edge_max)
        drift_vals = [nonlinear_drift[0], nonlinear_drift[1]]
    
        # Add some noise
        noise_dist_0 = pyro.distributions.Normal(loc = drift_vals[0], scale = sigma_cal)
        noise_dist_1 = pyro.distributions.Normal(loc = drift_vals[1], scale = sigma_cal)
        
        obs0_or_None = observations[0] if observations is not None else None 
        obs1_or_None = observations[1] if observations is not None else None
        
        noisy_obs_0 = pyro.sample('noise_0', noise_dist_0, bs = obs0_or_None)
        noisy_obs_1 = pyro.sample('noise_1', noise_dist_1, bs = obs1_or_None)
    
    output = [noisy_obs_0, noisy_obs_1]
    return output



"""
    4. Build guide
"""


# i) Build the guide

def guide(input_vars, observations = None):
    pass
    
    
    
    
    

# iii) illustrate model and guide

graphical_model = pyro.render_model(model = model, model_args= (input_vars,),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (input_vars,),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide

# iv) Record example outputs of model and guide prior to training

n_model_samples = 2
n_guide_samples = 100

predictive = pyro.infer.Predictive
prior_predictive_pretrain_dict = predictive(model, num_samples = n_model_samples)(input_vars)
posterior_pretrain_dict = predictive(guide, num_samples = n_guide_samples)(input_vars)
posterior_predictive_pretrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)(input_vars)





"""
    5. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 1})
elbo = pyro.infer.Trace_ELBO(num_particles = 1,
                                 max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (input_vars, observations_perstaff)
loss_sequence = []
for step in range(100):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
# iii) Record example outputs of model and guide post training

prior_predictive_posttrain_dict = predictive(model, num_samples = n_model_samples)(input_vars)
posterior_posttrain_dict = predictive(guide, num_samples = n_guide_samples)(input_vars)
posterior_predictive_posttrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)(input_vars)

for name, value in pyro.get_param_store().items():
    print(name, value)


"""
    6. Plots and illustration
"""




# i) Plot loss

plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# ii) Plot each group separately

# Some renamings for plotting
staff_types_reduced = staff_type_perjob
# Build minimal tensor of fitted alphas

def estimate_alpha(data):
    batched = data.ndim == 3
    if not batched:
        data = data.unsqueeze(0)  # → [1, n_obs, n_edge_max]
    alpha_tensor = torch.zeros(tuple(data.shape[:-1]) + (2,))
    
    for k in range(n_obs):
        rod_k =  staff_id_perjob[k]
        n_meas_rod_k = n_meas_perstaff[rod_k]
        n_edge_rod_k = n_edge_perstaff[rod_k]
            
        indices_obs_rod_k = torch.tensor(obsnr_perstaff[rod_k])
        x_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_rod_k)
        A_k = torch.vstack((torch.ones(n_edge_rod_k), x_rod_k)).T
        pinv_A = torch.linalg.pinv(A_k.T @ A_k) @ A_k.T  # shape [2, n_edge_rod_k]

        obs_k = data[:, k, :n_edge_rod_k]  # shape [n_samples, n_edge_rod_k]
        alpha_k = torch.einsum('ij,sj->si', pinv_A, obs_k)  # [n_samples, 2]

        alpha_tensor[:, k, :] = alpha_k
        
        # alpha_k = torch.linalg.pinv(A_k.T@A_k)@A_k.T@observations[k,:n_edge_rod_k]
        # alpha_tensor[k,:] = alpha_k
    return alpha_tensor

minimal_tensor = estimate_alpha(observations)


plt.figure(dpi=300)

for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[0,indices]
    plt.scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
plt.xlabel('Levelling rod offset [µm]')
plt.ylabel('Levelling rod scale [ppm]')
plt.title('Offset and scale of rods by staff type')
plt.legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# iii) Joint plot
# In this plot, we showcase the impact of training onto the parameter mu, which
# determines the prior distribution of the parameters per class.

offsets = minimal_tensor[:,:, 0].numpy()
scales = minimal_tensor[:,:, 1].numpy()
x_min = min(offsets.flatten()) - 10
x_max = max(offsets.flatten()) + 10
y_min = min(scales.flatten()) - 10
y_max = max(scales.flatten()) + 10

# Set up figure with 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

# Plot 1: Scatterplot
for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[0,indices]
    axs[0].scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
axs[0].set_xlabel('Levelling rod offset [µm]')
axs[0].set_ylabel('Levelling rod scale [ppm]')
axs[0].set_title('Offset and scale of rods by staff type')
axs[0].legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')


# Plot 2: 2D KDE (Pretraining)
prior_predictive_pretrain_tensor = build_tensor_from_dict(prior_predictive_pretrain_dict)
prior_predictive_pretrain_tensor_alpha = estimate_alpha(prior_predictive_pretrain_tensor[0,:,:])
prior_obs_dicts = build_dicts_from_data(prior_predictive_pretrain_tensor_alpha)
prior_class_obs_dict = prior_obs_dicts['data_dict_class']
prior_id_obs_dict = prior_obs_dicts['data_dict_id']
for class_name, tensor in prior_class_obs_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    offset = tensor[:, :, 0].flatten().numpy()
    scale = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=offset, y=scale, ax=axs[1],
        fill=True, bw_adjust=2, alpha = 0.3, label=class_name
    )

axs[1].set_title("Pretraining KDE by Class")
axs[1].set_xlabel("Offset [µm]")
axs[1].set_ylabel("Scale [ppm]")

# Plot 3: 2D KDE (Posttraining)
prior_predictive_posttrain_tensor = build_tensor_from_dict(prior_predictive_posttrain_dict)
prior_predictive_posttrain_tensor_alpha = estimate_alpha(prior_predictive_posttrain_tensor[0,:,:])
prior_obs_dicts = build_dicts_from_data(prior_predictive_posttrain_tensor_alpha)
prior_class_obs_dict = prior_obs_dicts['data_dict_class']
prior_id_obs_dict = prior_obs_dicts['data_dict_id']
for class_name, tensor in prior_class_obs_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    offset = tensor[:, :, 0].flatten().numpy()
    scale = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=offset, y=scale, ax=axs[2],
        fill=True, bw_adjust=2, alpha = 0.3, label=class_name
    )

axs[2].set_title("Posttraining KDE by Class")
axs[2].set_xlabel("Offset [µm]")
axs[2].set_ylabel("Scale [ppm]")
# axs[1].legend(title="Rod Type")

for ax in axs: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
plt.tight_layout()
plt.show()


# iii) Pre/ Posttrain prior
# Plot prior for different classes before and after training

def to_df(samples, cls_labels, label):
    """
    samples : Tensor [n_samp, n_obs, 2]
    cls_labels : list[str] length n_obs
    label : 'prior'|'post'
    """
    n_samp, n_obs, _ = samples.shape
    flat = samples.reshape(-1, 2)
    df = pd.DataFrame(flat, columns=["offset", "scale"])
    df["class"] = cls_labels * n_samp
    df["which"] = label
    return df

# raw observations' class labels
cls_labels_extended = [label for label in staff_type_perstaff for _ in range(n_meas_max)]#staff_type_perstaff*n_meas_max
tilt_labels_extended = [label for label in tilt_type_perstaff for _ in range(n_meas_max)]#staff_type_perstaff*n_meas_max

prior_predictive_pretrain_tensor = build_tensor_from_dict(prior_predictive_pretrain_dict)
prior_predictive_posttrain_tensor = build_tensor_from_dict(prior_predictive_posttrain_dict)
prior_predictive_pretrain_tensor_alpha = estimate_alpha(prior_predictive_pretrain_tensor)
prior_predictive_posttrain_tensor_alpha = estimate_alpha(prior_predictive_posttrain_tensor)
df_pretrain  = to_df(prior_predictive_pretrain_tensor_alpha,  cls_labels_extended, "pretrain")
df_posttrain   = to_df(prior_predictive_posttrain_tensor_alpha, cls_labels_extended, "posttrain")
df_long   = pd.concat([df_pretrain, df_posttrain], ignore_index=True)

g = sns.FacetGrid(
        df_long, col="class", row="which",
        height=3, aspect=1, despine=False
    )
g.map_dataframe(
        sns.kdeplot, x="offset", y="scale",
        fill=True, thresh=0.02, bw_adjust=1.2
    )
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.set(xlabel="Offset [µm]", ylabel="Scale [ppm]")
xmin, xmax = -200,  200    #  example numbers
ymin, ymax = -200,  200

for ax in g.axes.flat:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
plt.tight_layout()


# iv) Caterpillar plot shows posterior distribution for each rod

alpha_samples = posterior_posttrain_dict["alpha_rods"]    # [n_samp, n_ids, 2]
mean_off  = alpha_samples[:, :, 0].mean(0).numpy()   # [n_ids]
hdi_off   = az.hdi(alpha_samples[:, :, 0].numpy(), hdi_prob=0.95)  # [n_ids,2]

fig, ax = plt.subplots(figsize=(6, len(mean_off)*0.25))
ypos = np.arange(len(mean_off))

for i, (m, h) in enumerate(zip(mean_off, hdi_off)):
    cls_idx = id_class_indices[i]                 # class index for staff_id=i
    color   = type_to_color[index_to_type[int(cls_idx)]]
    ax.errorbar(
        x=m, y=i,
        xerr=[[m - h[0]], [h[1] - m]],
        fmt='o', color=color, capsize=3
    )

ax.set_yticks(ypos)
ax.set_yticklabels([f"id {i}" for i in ypos])
ax.set_xlabel("Offset [µm]")
ax.set_title("Posterior mean ± 95 % HPDI per rod")
plt.tight_layout()


# v) Arrow plot shows the difference between trained prior and the posterior

alpha_mean = posterior_posttrain_dict["alpha_rods"].mean(0)   # [n_ids,2]
mu_learned = pyro.param("mu_alpha_prod").detach()

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(mu_learned[:,0], mu_learned[:,1], c='k', marker='s', label='class µ')

for sid in range(n_ids):
    cls   = id_class_indices[sid]
    color = type_to_color[index_to_type[int(cls)]]
    ax.arrow(mu_learned[cls,0], mu_learned[cls,1],
             alpha_mean[sid,0] - mu_learned[cls,0],
             alpha_mean[sid,1] - mu_learned[cls,1],
             head_width=3, length_includes_head=True,
             color=color, alpha=.6)

ax.set_xlabel("Offset [µm]")
ax.set_ylabel("Scale [ppm]")
ax.set_title("Each rod’s posterior mean α̂ vs. its class centre µ")
plt.tight_layout(); plt.show()




# vi) Plot with learned priors

# Plot raw observations, distinguished by corresponding staff class
plt.figure(dpi=300)

for stype in unique_types:
    idx   = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    pts   = minimal_tensor[:, idx,:]
    plt.scatter(pts[:, :, 0], pts[:,:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])

# Plot mu and sigma prior learned from data
mu_hat     = pyro.param("mu_alpha_prod").detach()          # [n_classes, 2]
Sigma_hat  = pyro.param("Sigma_alpha_prod").detach()       # [n_classes, 2, 2]

for i, cls in enumerate(unique_types):
    plt.scatter(mu_hat[i,0], mu_hat[i,1],
                marker='X', s=90, lw=1.5,
                color=type_to_color[cls],
                label='mu ' + cls)   

# Plot 95 % confidence ellipses
CHI2_95 = 5.991  # χ²_{2,0.95}

def add_cov_ellipse(ax, mean, cov, color, **kwargs):
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]          # big → small
    vals, vecs = vals[order], vecs[:, order]

    # 95 % χ² quantile for 2 dof
    width, height = 2 * np.sqrt(vals * CHI2_95)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))  # major-axis vector

    e = Ellipse(mean, width, height, angle,
                facecolor='none', edgecolor=color,
                linewidth=2, **kwargs)
    ax.add_patch(e)

ax = plt.gca()
for cls in range(n_types):
    add_cov_ellipse(ax,
                    mean=mu_hat[cls].numpy(),
                    cov =Sigma_hat[cls].numpy(),
                    color=type_to_color[index_to_type[cls]],
                    ls='--')

# Adjust some details
plt.xlabel('Levelling-rod offset [µm]')
plt.ylabel('Levelling-rod scale [ppm]')
plt.title('Offset & scale: observations  |  posterior μ̂  |  95 % ellipse')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
xmin, xmax = -100,  100    #  example numbers
ymin, ymax = -50,  150

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()



# vii) Plot posterior samples of parameters

plt.figure(dpi=300)
# Plot samples from the alpha posterior
alpha_samples = posterior_posttrain_dict["alpha_rods"]      # [n_samp, n_ids, 2]

# choose how many samples per ID to plot (thin for readability)
NSHOW = 100
idx_show = torch.randperm(alpha_samples.shape[0])[:NSHOW]

alpha_thin = alpha_samples[idx_show]                   # [NSHOW, n_ids, 2]
alpha_thin = alpha_thin.reshape(-1, 2).numpy()         # [NSHOW*n_ids, 2]

# we need matching colours: repeat each id’s colour NSHOW times
colours = []
for sid in range(n_ids):
    cls   = id_class_indices[sid].item()
    c     = type_to_color[index_to_type[cls]]
    colours.extend([c]*NSHOW)

plt.scatter(alpha_thin[:,0], alpha_thin[:,1],
            s=20, alpha=.1, marker='o',
            edgecolors='none',
            # c=colours,        # Something not right with color classes
            label='posterior α samples')

# Plot raw observations, distinguished by corresponding staff class
for stype in unique_types:
    idx   = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    pts   = minimal_tensor[:, idx,:]
    plt.scatter(pts[:, :, 0], pts[:,:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])
    
# Plotting adjustments
plt.xlabel('Levelling-rod offset [µm]')
plt.ylabel('Levelling-rod scale [ppm]')
plt.title('Observed data  |  learned class μ  |  posterior α samples')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

xmin, xmax = -100, 100
ymin, ymax = -50, 50
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.show()


Sigma_cal_pyro = pyro.get_param_store()['sigma_cal']
print('Estimated edge uncertainty : ', Sigma_cal_pyro)
# print('Sigma offset calibration: ', torch.sqrt(Sigma_cal_pyro[0,0]) )
# print('Sigma scale calibration: ', torch.sqrt(Sigma_cal_pyro[1,1]) )


# viii) Plot calibration covariance

# plt.figure(dpi = 300)
# plt.imshow(Sigma_cal_pyro.detach().numpy())
# plt.title('Error covariance matrix calibration')

# ix) Plot edge observations per class for data, pretrain model, posttrain model

# Build posterior predictive distributions
posterior_predictive_pretrain_tensor = build_tensor_from_dict(posterior_predictive_pretrain_dict)
posterior_predictive_posttrain_tensor = build_tensor_from_dict(posterior_predictive_posttrain_dict)

# mask the nans
n_obs_extended  = len(cls_labels_extended)
rod_k_extended_obs = [staff_id for staff_id in staff_list for _ in range(n_meas_max)]
posterior_predictive_pretrain_tensor_nan = torch.full(posterior_predictive_pretrain_tensor.shape, float("nan"))
posterior_predictive_posttrain_tensor_nan = torch.full(posterior_predictive_posttrain_tensor.shape, float("nan"))
for k in range(n_obs_extended):
    rod_k = rod_k_extended_obs[k]
    n_edge_rod_k = n_edge_perstaff[rod_k]
    
    indices_obs_rod_k = torch.tensor(obsnr_perstaff[rod_k])
    x_staff_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
    
    posterior_predictive_pretrain_tensor_nan[:, k, :n_edge_rod_k] = posterior_predictive_pretrain_tensor[:, k, :n_edge_rod_k]
    posterior_predictive_posttrain_tensor_nan[:, k, :n_edge_rod_k] = posterior_predictive_posttrain_tensor[:, k, :n_edge_rod_k]

def plot_data_model_grid(cls_labels_extended, observations,
                         predictive_pre, predictive_post,
                         n_types_to_plot=None, max_lines_per_cell=10):

    n_obs, n_edge = observations.shape
    n_samples = predictive_post.shape[0]

    # Unique rod types
    unique_classes = list(set(cls_labels_extended))
    if n_types_to_plot is not None:
        unique_classes = unique_classes[:n_types_to_plot]
    n_types = len(unique_classes)

    fig, axs = plt.subplots(3, n_types, figsize=(4 * n_types, 8), sharey=True, sharex=True)
    fig.suptitle("Edge observations from data and model", fontsize=16)
    if n_types == 1:
        axs = axs[:, None]  # make it 2D if only one type

    for col, rod_type in enumerate(unique_classes):
        idxs = [i for i, lbl in enumerate(cls_labels_extended) if lbl == rod_type][:max_lines_per_cell]

        # row 0: raw data
        for i in idxs:
            axs[0, col].plot(observations_extended[i].cpu(), color='black', alpha=0.6)

        # row 1: pretrain predictive mean
        pred_mean_pre = predictive_pre[:, idxs, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[1, col].plot(pred_mean_pre[i].cpu(), color='blue', alpha=0.6)

        # row 2: posttrain predictive mean
        pred_mean_post = predictive_post[:, idxs, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[2, col].plot(pred_mean_post[i].cpu(), color='green', alpha=0.6)

        axs[0, col].set_title(f"Rod Type {rod_type}")


    axs[0, 0].set_ylabel("Data")
    axs[1, 0].set_ylabel("Pretrain")
    axs[2, 0].set_ylabel("Posttrain")

    for ax in axs.flatten():
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
plot_data_model_grid(cls_labels_extended,
                 observations,
                 posterior_predictive_pretrain_tensor,
                 posterior_predictive_posttrain_tensor,
                 n_types_to_plot=5, max_lines_per_cell = 20)

    
    

# x) Plot nonlinear trends for different tilt classes

# Extend the labels and set -1 where no measurements
dummy_label = -1  # or 'dummy' if you prefer strings

tilt_labels_extended = []

for per_staff_list in tilt_type_perstaff:
    types = [t.item() for t in per_staff_list]
    padded = types + [dummy_label] * (n_meas_max - len(types))
    tilt_labels_extended.extend(padded)
# tilt_labels_extended = [t.item() for sublist in tilt_type_perstaff for t in sublist]


def plot_tilt_model_grid(tilt_labels_extended, observations,
                          predictive_pre, predictive_post,
                          n_types_to_plot=None, max_lines_per_cell=10,
                          suptitle=None):
    """
    Plots a 3-row grid:
        Row 1: Observations
        Row 2: Pretrain predictive mean
        Row 3: Posttrain predictive mean

    Grouped by unique tilt types.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure tilt_labels_extended is a numpy array for fast comparison
    tilt_labels_extended = np.array(tilt_labels_extended)
    n_obs, n_edge = observations.shape
    # n_samples = predictive_post.shape[0]

    unique_tilts = list(np.unique(tilt_labels_extended))
    if n_types_to_plot is not None:
        unique_tilts = unique_tilts[:n_types_to_plot]
    dummy_label = -1
    unique_tilts = [t for t in unique_tilts if t != dummy_label]
    n_tilts = len(unique_tilts)

    fig, axs = plt.subplots(3, n_tilts, figsize=(4 * n_tilts, 8), sharey=True, sharex=True)

    if n_tilts == 1:
        axs = axs[:, None]

    for col, tilt_type in enumerate(unique_tilts):
        mask = (tilt_labels_extended == tilt_type)
        idxs = np.where(mask)[0][:max_lines_per_cell]
        idxs_torch = torch.tensor(idxs, device=observations.device)

        # Row 0: observations
        for i in idxs:
            axs[0, col].plot(observations[i].cpu(), color='black', alpha=0.6)

        # Row 1: pretrain mean
        pred_mean_pre = predictive_pre[:, idxs_torch, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[1, col].plot(pred_mean_pre[i].cpu(), color='blue', alpha=0.6)

        # Row 2: posttrain mean
        pred_mean_post = predictive_post[:, idxs_torch, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[2, col].plot(pred_mean_post[i].cpu(), color='green', alpha=0.6)

        axs[0, col].set_title(f"Tilt Type {tilt_type}")

    axs[0, 0].set_ylabel("Data")
    axs[1, 0].set_ylabel("Pretrain")
    axs[2, 0].set_ylabel("Posttrain")

    for ax in axs.flatten():
        ax.grid(True)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout()
    plt.show()

plot_tilt_model_grid(tilt_labels_extended,
                      observations_extended,
                      posterior_predictive_pretrain_tensor,
                      posterior_predictive_posttrain_tensor,
                      suptitle="Nonlinear Trends by Tilt Type")


# xi) Plot tilt_ann models
x_rod_length = torch.linspace(0,1,100).reshape([100,1])
type_0_tilt_impact = tilt_ann.models[0](x_rod_length).detach().numpy()
type_1_tilt_impact = tilt_ann.models[1](x_rod_length).detach().numpy()
type_2_tilt_impact = tilt_ann.models[2](x_rod_length).detach().numpy()

plt.figure(figsize=(6, 4))

plt.plot(type_0_tilt_impact, label="Tilt Type 0")
plt.plot(type_1_tilt_impact, label="Tilt Type 1")
plt.plot(type_2_tilt_impact, label="Tilt Type 2")

plt.xlabel("Edge Index")
plt.ylabel("Tilt Effect")
plt.title("Tilt Function Outputs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
