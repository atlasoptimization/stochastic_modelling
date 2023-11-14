""" 
The goal of this script is to perform simulation and inference using pyro.
We will use a synthetic timeseries and then train a simple model to recreate
the behavior of the ynthetic timeseries generated by an (unknown) process.
For this, do the following:
    1. Definitions and imports
    2. Build stochastic model
    3. Inference
    4. Plots and illustrations
"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import seaborn as sns
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
from pyro.infer import SVI, Trace_ELBO
import torch


# ii) Definitions

n_t = 24
n_simu = 200

t = np.linspace(0, n_t, n_t)



"""
    2. Generate data ---------------------------------------------------------
"""


# Original stochastic model (assumed unknown alater on)

mu = 10*np.sin((1*np.pi/n_t)*t)
d = 3
cov_fun = lambda s,t: np.exp(-((t-s)/d)**2)
# cov_fun = lambda s,t: np.cos((2*np.pi/n_t)*(t-s))

cov_mat = np.zeros([n_t,n_t])
for k in range(n_t):
    for l in range(n_t):
        cov_mat[k,l] = cov_fun(t[k],t[l])

x_simu = torch.zeros([n_simu, n_t])

for k in range(n_simu):
    x_simu[k,:] = torch.tensor(np.random.multivariate_normal(mu, cov_mat))



"""
    2. Build stochastic model
"""


# i) Initializations

pyro.clear_param_store()



#  ii) SVI Model for simulations and inference 

def stoch_model_svi(x_simu = None):                
    mu = pyro.param('mu', 0.5*torch.ones([n_t]))
    sigma = pyro.param('sigma', 0.5*torch.eye(n_t),constraint=pyro.distributions.constraints.positive_definite)
    # sigma = 0.01*torch.eye(n_t)
    
    with pyro.plate('x_simu',n_simu):
        sample = pyro.sample('obs', dist.MultivariateNormal(loc = mu, covariance_matrix = sigma +0.01*torch.eye(n_t)),obs = x_simu)
        return sample
    # pyro plate works as follows:
    #   the second argument tells plate to copy the subsequent sample definition n_simu times
    #   then
    
    # The following does the same. It is more interpretable but much slower and requires the existence of data.
    # for k in range(x_simu.shape[0]):
    #     pyro.sample("obs_{}".format(k), dist.MultivariateNormal(loc = mu, covariance_matrix = sigma+0.01*torch.eye(n_t)), obs = x_simu[k,:])

    
def stoch_guide(y_obs=None):
    pass


# ii) Do svi inference

lr=0.005
n_steps=10000

adam_params = {"lr": lr}
adam = pyro.optim.Adam(adam_params)
svi = SVI(stoch_model_svi, stoch_guide, adam, loss=Trace_ELBO())

for step in range(n_steps):
    # What was going on with x_simu = [n_t,n_simu]?
    #svi_step(x_simu[:,0]) : takes only the first dataset
    #svi_step(x_simu) : takes only the first dataset
    #svi_step(x_simu[0,:]) : takes only the first dataset
    loss = svi.step(x_simu)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

mle_estimate_mu = pyro.param("mu")
# mle_estimate_sigma = pyro.param("sigma")

mle_estimate_mu_numpy= pyro.param("mu").detach().numpy()
mle_estimate_sigma_numpy = pyro.param("sigma").detach().numpy()
# mle_estimate_sigma_numpy = 0.01*torch.eye(n_t)

# synth_simu = pyro.sample("synth" ,dist.MultivariateNormal(loc = mle_estimate_mu, covariance_matrix = mle_estimate_sigma))
synth_simu = np.random.multivariate_normal(mean = mle_estimate_mu_numpy, cov = mle_estimate_sigma_numpy + 0.01*np.eye(n_t))



import matplotlib.pyplot as plt
plt.figure(1, dpi = 300)
plt.plot(x_simu.T)
plt.plot(synth_simu, color = 'r', linestyle = '--', linewidth = 3)
plt.plot(mle_estimate_mu_numpy, color = 'k', linestyle = '--', linewidth = 3)
plt.figure(2, dpi = 300)
plt.imshow(mle_estimate_sigma_numpy)



# # iii) Evaluate ppd

ppd=pyro.infer.Predictive(stoch_model_svi,guide=stoch_guide, num_samples=1)
svi_samples=ppd()
svi_obs=svi_samples["obs"]




# """
#     4. Plots and illustrations
# """


# # data=y_obs
# # def train(model, guide, lr=0.005, n_steps=1001):
# #     pyro.clear_param_store()
# #     adam_params = {"lr": lr}
# #     adam = pyro.optim.Adam(adam_params)
# #     svi = SVI(model, guide, adam, loss=Trace_ELBO())

# #     for step in range(n_steps):
# #         loss = svi.step(data)
# #         if step % 50 == 0:
# #             print('[iter {}]  loss: {:.4f}'.format(step, loss))
            
            
            
# # def model_mle(data):
# #     # note that we need to include the interval constraint;
# #     # in original_model() this constraint appears implicitly in
# #     # the support of the Beta distribution.
# #     f = pyro.param("latent_fairness", torch.tensor(0.5),
# #                    constraint=dist.constraints.unit_interval)
# #     with pyro.plate("data", data.size(0)):
# #         pyro.sample("obs", dist.Bernoulli(f), obs=data)
        
        
# # def guide_mle(data):
# #     pass

# # train(model_mle, guide_mle)

# # mle_estimate = pyro.param("latent_fairness").item()
# # print("Our MLE estimate of the latent fairness is {:.3f}".format(mle_estimate))




















