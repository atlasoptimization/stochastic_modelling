#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to emulate how a simple linear relationship gets distorted 
into something nonlinear and noisy when relevant relations are ignored. In this
sequence of investigations, we will incrementally introduce mismatch between a
generative model and visualization of results showcasing how improperly handled
input variables lead to nonlinearity and the impression of noise. 
Complicated behavior might therefore follow from misspecified assumptions rather
than from true functional behavior or randomness.
We showcase multiple models:
    F = Functional relation is always linear, e.g. f,g linear in the input vars
    Var = Input variables can be x or x,y
    Ran = Which parts of the model are random, can be x, t with x(t),y(t)  x,y, and F
These lead to multiple possible combinations we will investigate.
To analyse these relations, we will do the following:
    1. Imports and definitions
    2. Simulate cases 1, 2
    3. Simulate cases 3, 4, 5
    4. Simulate cases 6, 7
    5. Plots and illustrations
The specific cases and what can be inferred from them are listed down below.
    
We use the following abbreviations: 
    F = Functional model for f(x,y) g(xy)
    L + Linear model (we only use linear)
    V = Variables ( either f,g, depend only on X or on (X,Y))
    R = Random (Multiple possibilities: X random, X,Y Random, F = Function values
                random, i.e. noisy, T random and X,Y = NL(T))
    NL = Nonlinear function

Case 1
F,V,R = [L, x, x]    
Generative model maximally simple : f = alpha*x, g = beta*x, sample x
    Then p(f,g) is trivial with f = phi(g) = (alpha/beta)x
    Consequently f and g show a linear relation
   
Case 2
F,V,R = [L, x, (x,F)]    
Generative model with noise : f = alpha*x + eps_1, g = beta*x + eps_2, 
                                sample x, sample eps
    Then p(f,g) is simple with f = (alpha/beta)x + noise
    Consequently f and g show a noisy linear relation

Case 3
F,V,R = [L, (x,y) , (x(t), y(t))]
Generative model with x,y covarying : f = x,
                                      g = y, 
                                sample t -> (x(t), y(t))
    Then since there is a single random variable t determining a path x(t), y(t),
    x,y always vary together in a nonlinear way leading to nonlinearity in f,g
    This nonlinearity is a bit surprising since f,g are so simple as the source
    of the nonlinearity is linear model and a path through it.

Case 4
F,V,R = [L, (x,y) , (x(t), y(t))]
Generative model with x,y covarying : f = alpha_1*x + alpha_2*y,
                                      g = beta_1*x + beta_2*y, 
                                sample t -> (x(t), y(t))
    Very similar to case 3 in terms of construction but now f,g depend on x and y.
    This situation is akin to x,y both depending on an unaccounted for third
    factor. They individually vary and their variation is coupled.

Case 5
F,V,R = [L, (x,y) , (x(t), y(t), F)]
Generative model with x,y covarying and noise : 
                                      f = alpha_1*x + alpha_2*y + eps_1,
                                      g = beta_1*x + beta_2*y + eps_2, 
                                sample t -> (x(t), y(t)), sample eps
    This is very similar to case 3 in terms of construction but with noise added
    on top of the observations. The result is f,g varying nonlinearly and noisily 
    with the underlying pattern hard to distinguish.
    
Case 6
F,V,R = [L, (x,y) , (x, y)]
Generative model with x,y random : 
                                      f = alpha_1*x + alpha_2*y ,
                                      g = beta_1*x + beta_2*y , 
                                sample (x, y)
    This looks insterestingly different from the case where the variation in x,y
    is explainable by a single variable introducing a nonlinear pattern. The results
    simply look more disordered with less structure visible
    
Case 7
F,V,R = [L, (x,y) , (x,y F)]
Generative model with x,y random and noise : 
                                      f = alpha_1*x + alpha_2*y + eps_1,
                                      g = beta_1*x + beta_2*y + eps_2, 
                                sample (x,y)), sample eps
    No qualititative difference in visualization compared to case 5. The relationships
    are lost to noise even more heavily.
    
    
"""
    
"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_pts = 500
unit = [0,1]
sigma_noise = 0.3

alpha = 1
beta = 1

alpha_1 = 1
alpha_2 = 1
beta_1 = 2
beta_2 = 0.5

# Nonlinearities
f_nl_x = lambda x: x
# f_nl_x = lambda x: np.cos(2*np.pi*x)
# f_nl_x = lambda x: np.cos(15*np.pi*x)
# f_nl_x = lambda x: np.exp(-x)
# f_nl_x = lambda y: np.arctan(4*np.pi*x)

# f_nl_y = lambda y: y
# f_nl_y = lambda y: np.cos(2*np.pi*y)
# f_nl_y = lambda y: np.cos(14*np.pi*y)
# f_nl_y = lambda y: np.exp(-y)
f_nl_y = lambda y: np.arctan(4*np.pi*y)


"""
    2. Simulate cases 1, 2
"""


# i) Case 1
# F,V,R = [L, x, x]    
# Generative model maximally simple : f = alpha*x, g = beta*x, sample x
#     Then p(f,g) is trivial with f = phi(g) = (alpha/beta)x
#     Consequently f and g show a linear relation

x_c1 = np.random.uniform(unit[0], unit[1], size = n_pts)

f_c1 = alpha*x_c1
g_c1 = beta*x_c1


# ii) Case 2
# F,V,R = [L, x, (x,F)]    
# Generative model with noise : f = alpha*x + eps_1, g = beta*x + eps_2, 
#                                 sample x, sample eps
#     Then p(f,g) is simple with f = (alpha/beta)x + noise
#     Consequently f and g show a noisy linear relation

x_c2 = np.random.uniform(unit[0], unit[1], size = n_pts)

noise_f_c2 = np.random.normal(0, sigma_noise, size = n_pts)
noise_g_c2 = np.random.normal(0, sigma_noise, size = n_pts)

f_c2 = alpha*x_c2 + noise_f_c2
g_c2 = beta*x_c2 + noise_g_c2


"""
    3. Simulate cases 3, 4, 5
"""


# i) Case 3
# F,V,R = [L, (x,y) , (x(t), y(t))]
# Generative model with x,y covarying : f = x,
#                                       g = y, 
#                                 sample t -> (x(t), y(t))
#     Then since there is a single random variable t determining a path x(t), y(t),
#     x,y always vary together in a nonlinear way leading to nonlinearity in f,g
#     This nonlinearity is a bit surprising since f,g are so simple as the source
#     of the nonlinearity is linear model and a path through it.

t_c3 = np.random.uniform(unit[0], unit[1], size = n_pts)

x_c3 = f_nl_x(t_c3)
y_c3 = f_nl_y(t_c3)

f_c3 = x_c3 
g_c3 = y_c3


# ii) Case 4
# F,V,R = [L, (x,y) , (x(t), y(t))]
# Generative model with x,y covarying : f = alpha_1*x + alpha_2*y,
#                                       g = beta_1*x + beta_2*y, 
#                                 sample t -> (x(t), y(t))
#     Very similar to case 3 in terms of construction but now f,g depend on x and y.
#     This situation is akin to x,y both depending on an unaccounted for third
#     factor. They individually vary and their variation is coupled.

t_c4 = np.random.uniform(unit[0], unit[1], size = n_pts)

x_c4 = f_nl_x(t_c4)
y_c4 = f_nl_y(t_c4)

f_c4 = alpha_1*x_c4 + alpha_2*y_c4
g_c4 = beta_1*x_c4 + beta_2*y_c4


# ii) Case 5
# F,V,R = [L, (x,y) , (x(t), y(t), F)]
# Generative model with x,y covarying and noise : 
#                                       f = alpha_1*x + alpha_2*y + eps_1,
#                                       g = beta_1*x + beta_2*y + eps_2, 
#                                 sample t -> (x(t), y(t)), sample eps
#     This is very similar to case 3 in terms of construction but with noise added
#     on top of the observations. The result is f,g varying nonlinearly and noisily 
#     with the underlying pattern hard to distinguish.

t_c5 = np.random.uniform(unit[0], unit[1], size = n_pts)

x_c5 = f_nl_x(t_c5)
y_c5 = f_nl_y(t_c5)

noise_f_c5 = np.random.normal(0, sigma_noise, size = n_pts)
noise_g_c5 = np.random.normal(0, sigma_noise, size = n_pts)

f_c5 = alpha_1*x_c5 + alpha_2*y_c5 + noise_f_c5
g_c5 = beta_1*x_c5 + beta_2*y_c5 + noise_g_c5


"""
    4. Simulate cases 6, 7
"""


# i) Case 6
# F,V,R = [L, (x,y) , (x, y)]
# Generative model with x,y random : 
#                                       f = alpha_1*x + alpha_2*y ,
#                                       g = beta_1*x + beta_2*y , 
#                                 sample (x, y)
#     This looks insterestingly different from the case where the variation in x,y
#     is explainable by a single variable introducing a nonlinear pattern. The results
#     simply look more disordered with less structure visible


x_c6 = np.random.uniform(unit[0], unit[1], size = n_pts)
y_c6 =np.random.uniform(unit[0], unit[1], size = n_pts)

f_c6 = alpha_1*x_c6 + alpha_2*y_c6
g_c6 = beta_1*x_c6 + beta_2*y_c6


# ii) Case 7
# F,V,R = [L, (x,y) , (x,y F)]
# Generative model with x,y random and noise : 
#                                       f = alpha_1*x + alpha_2*y + eps_1,
#                                       g = beta_1*x + beta_2*y + eps_2, 
#                                 sample (x,y)), sample eps
#     No qualititative difference in visualization compared to case 5. The relationships
#     are lost to noise even more heavily.
    
x_c7 = np.random.uniform(unit[0], unit[1], size = n_pts)
y_c7 = np.random.uniform(unit[0], unit[1], size = n_pts)

noise_f_c7 = np.random.normal(0, sigma_noise, size = n_pts)
noise_g_c7 = np.random.normal(0, sigma_noise, size = n_pts)

f_c7 = alpha_1*x_c7 + alpha_2*y_c7 + noise_f_c7
g_c7 = beta_1*x_c7 + beta_2*y_c7 + noise_g_c7



"""
    5. Plots and illustrations
"""


# i) Case 1 plot f,g ; f,x ; g,x

fig1, axs1 = plt.subplots(1, 3, dpi = 300, figsize = (15,5))
fig1.suptitle('Case 1: F,V,R = [L, x, x] \n \n f = alpha * x \n g = beta * x')
axs1[0].set_title(' g vs f')
axs1[0].scatter(f_c1, g_c1)
axs1[0].set_xlabel('Function f')
axs1[0].set_ylabel('Function g')

axs1[1].set_title(' f vs x')
axs1[1].scatter(x_c1, f_c1)
axs1[1].set_xlabel('Variable x')
axs1[1].set_ylabel('Function f')

axs1[2].set_title(' g vs x')
axs1[2].scatter(x_c1, g_c1)
axs1[2].set_xlabel('Variable x')
axs1[2].set_ylabel('Function g')

plt.tight_layout()
plt.show()


# ii) Case 2 plot f,g ; f,x ; g,x

fig2, axs2 = plt.subplots(1, 3, dpi = 300, figsize = (15,5))
fig2.suptitle('Case 2: F,V,R = [L, x, (x,F)] \n \n f = alpha * x + noise' 
              '\n g = beta * x + noise')
axs2[0].set_title(' g vs f')
axs2[0].scatter(f_c2, g_c2)
axs2[0].set_xlabel('Function f')
axs2[0].set_ylabel('Function g')

axs2[1].set_title(' f vs x')
axs2[1].scatter(x_c2, f_c2)
axs2[1].set_xlabel('Variable x')
axs2[1].set_ylabel('Function f')

axs2[2].set_title(' g vs x')
axs2[2].scatter(x_c2, g_c2)
axs2[2].set_xlabel('Variable x')
axs2[2].set_ylabel('Function g')

plt.tight_layout()
plt.show()


# iii) Case 3 plot f,g ; f,x ; g,x

fig3, axs3 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig3.suptitle('Case 3: F,V,R = [L, (x,y), (x(t), y(t))]  \n \n f = x' 
              '\n g = y \n t ~ U[0,1] \n x = t, y = nl(t)')
axs3[0,0].set_title(' g vs f')
axs3[0,0].scatter(f_c3, g_c3)
axs3[0,0].set_xlabel('Function f')
axs3[0,0].set_ylabel('Function g')

axs3[0,1].set_title(' f vs x')
axs3[0,1].scatter(x_c3, f_c3)
axs3[0,1].set_xlabel('Variable x')
axs3[0,1].set_ylabel('Function f')

axs3[0,2].set_title(' g vs x')
axs3[0,2].scatter(x_c3, g_c3)
axs3[0,2].set_xlabel('Variable x')
axs3[0,2].set_ylabel('Function g')

axs3[1,0].set_title(' y vs x')
axs3[1,0].scatter(x_c3, y_c3)
axs3[1,0].set_xlabel('Variable x')
axs3[1,0].set_ylabel('Variable y')

axs3[1,1].set_title(' f vs y')
axs3[1,1].scatter(y_c3, f_c3)
axs3[1,1].set_xlabel('Variable y')
axs3[1,1].set_ylabel('Function f')

axs3[1,2].set_title(' g vs y')
axs3[1,2].scatter(y_c3, g_c3)
axs3[1,2].set_xlabel('Variable y')
axs3[1,2].set_ylabel('Function g')


plt.tight_layout()
plt.show()


# iv) Case 4 plot f,g ; f,x ; g,x

fig4, axs4 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig4.suptitle('Case 4: F,V,R = [L, (x,y), (x(t), y(t))] \n \n f = alpha_1 * x + alpha_2 * y' 
              '\n g = beta_1 * x + beta_2 * y \n \n t ~ U[0,1] \n x = t, y = nl(t)')
axs4[0,0].set_title(' g vs f')
axs4[0,0].scatter(f_c4, g_c4)
axs4[0,0].set_xlabel('Function f')
axs4[0,0].set_ylabel('Function g')

axs4[0,1].set_title(' f vs x')
axs4[0,1].scatter(x_c4, f_c4)
axs4[0,1].set_xlabel('Variable x')
axs4[0,1].set_ylabel('Function f')

axs4[0,2].set_title(' g vs x')
axs4[0,2].scatter(x_c4, g_c4)
axs4[0,2].set_xlabel('Variable x')
axs4[0,2].set_ylabel('Function g')

axs4[1,0].set_title(' y vs x')
axs4[1,0].scatter(x_c4, y_c4)
axs4[1,0].set_xlabel('Variable x')
axs4[1,0].set_ylabel('Variable y')

axs4[1,1].set_title(' f vs y')
axs4[1,1].scatter(y_c4, f_c4)
axs4[1,1].set_xlabel('Variable y')
axs4[1,1].set_ylabel('Function f')

axs4[1,2].set_title(' g vs y')
axs4[1,2].scatter(y_c4, g_c4)
axs4[1,2].set_xlabel('Variable y')
axs4[1,2].set_ylabel('Function g')


plt.tight_layout()
plt.show()


# v) Case 5 plot f,g ; f,x ; g,x

fig5, axs5 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig5.suptitle('Case 5: F,V,R = [L, (x,y) , (x(t), y(t), F)]\n \n f = alpha_1 * x + alpha_2 * y + noise' 
              '\n g = beta_1 * x + beta_2 * y + noise \n \n noise ~ N(0,0.3) \n  t ~ U[0,1] \n x = t, y = nl(t)')
axs5[0,0].set_title(' g vs f')
axs5[0,0].scatter(f_c5, g_c5)
axs5[0,0].set_xlabel('Function f')
axs5[0,0].set_ylabel('Function g')

axs5[0,1].set_title(' f vs x')
axs5[0,1].scatter(x_c5, f_c5)
axs5[0,1].set_xlabel('Variable x')
axs5[0,1].set_ylabel('Function f')

axs5[0,2].set_title(' g vs x')
axs5[0,2].scatter(x_c5, g_c5)
axs5[0,2].set_xlabel('Variable x')
axs5[0,2].set_ylabel('Function g')

axs5[1,0].set_title(' y vs x')
axs5[1,0].scatter(x_c5, y_c5)
axs5[1,0].set_xlabel('Variable x')
axs5[1,0].set_ylabel('Variable y')

axs5[1,1].set_title(' f vs y')
axs5[1,1].scatter(y_c5, f_c5)
axs5[1,1].set_xlabel('Variable y')
axs5[1,1].set_ylabel('Function f')

axs5[1,2].set_title(' g vs y')
axs5[1,2].scatter(y_c5, g_c5)
axs5[1,2].set_xlabel('Variable y')
axs5[1,2].set_ylabel('Function g')


plt.tight_layout()
plt.show()


# vi) Case 6 plot f,g ; f,x ; g,x

fig6, axs6 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig6.suptitle('Case 6: F,V,R = [L, (x,y) , (x, y)]] \n \n f = alpha_1 * x + alpha_2 * y ' 
              '\n g = beta_1 * x + beta_2 * y \n \n x ~ U[0,1],  y = U[0,1]')
axs6[0,0].set_title(' g vs f')
axs6[0,0].scatter(f_c6, g_c6)
axs6[0,0].set_xlabel('Function f')
axs6[0,0].set_ylabel('Function g')

axs6[0,1].set_title(' f vs x')
axs6[0,1].scatter(x_c6, f_c6)
axs6[0,1].set_xlabel('Variable x')
axs6[0,1].set_ylabel('Function f')

axs6[0,2].set_title(' g vs x')
axs6[0,2].scatter(x_c6, g_c6)
axs6[0,2].set_xlabel('Variable x')
axs6[0,2].set_ylabel('Function g')

axs6[1,0].set_title(' y vs x')
axs6[1,0].scatter(x_c6, y_c6)
axs6[1,0].set_xlabel('Variable x')
axs6[1,0].set_ylabel('Variable y')

axs6[1,1].set_title(' f vs y')
axs6[1,1].scatter(y_c6, f_c6)
axs6[1,1].set_xlabel('Variable y')
axs6[1,1].set_ylabel('Function f')

axs6[1,2].set_title(' g vs y')
axs6[1,2].scatter(y_c6, g_c6)
axs6[1,2].set_xlabel('Variable y')
axs6[1,2].set_ylabel('Function g')


plt.tight_layout()
plt.show()


# vii) Case 7 plot f,g ; f,x ; g,x

fig7, axs7 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig7.suptitle('Case 7: F,V,R = [L, (x,y) , (x,y F)\n \n f = alpha_1 * x + alpha_2 * y + noise' 
              '\n g = beta_1 * x + beta_2 * y + noise \n \n noise ~ N(0,0.3) \n x ~ U[0,1],  y = U[0,1]')
axs7[0,0].set_title(' g vs f')
axs7[0,0].scatter(f_c7, g_c7)
axs7[0,0].set_xlabel('Function f')
axs7[0,0].set_ylabel('Function g')

axs7[0,1].set_title(' f vs x')
axs7[0,1].scatter(x_c7, f_c7)
axs7[0,1].set_xlabel('Variable x')
axs7[0,1].set_ylabel('Function f')

axs7[0,2].set_title(' g vs x')
axs7[0,2].scatter(x_c7, g_c7)
axs7[0,2].set_xlabel('Variable x')
axs7[0,2].set_ylabel('Function g')

axs7[1,0].set_title(' y vs x')
axs7[1,0].scatter(x_c7, y_c7)
axs7[1,0].set_xlabel('Variable x')
axs7[1,0].set_ylabel('Variable y')

axs7[1,1].set_title(' f vs y')
axs7[1,1].scatter(y_c7, f_c7)
axs7[1,1].set_xlabel('Variable y')
axs7[1,1].set_ylabel('Function f')

axs7[1,2].set_title(' g vs y')
axs7[1,2].scatter(y_c7, g_c7)
axs7[1,2].set_xlabel('Variable y')
axs7[1,2].set_ylabel('Function g')


plt.tight_layout()
plt.show()


# Remark: Assume that f,g are observed values and we do not have access to the
# underlying input variables x,y. They are hidden latents that we simply know are
# impacting the values for f,g. 
# When only f,g, are observed and compared to each other, their relationship can
# become arbitrarily nonlinear even though f,g both linearly depend on the latents
# x,y, simply by adjusting the sampling mechanics for x,y. Correlation between
# x,y, becomes nonlinearity in the graph (f,g) and randomness in x,y becomes 
# additional volume in (f,g). Depending on the x,y sweeping pattern, the result
# can look like noise in the graph (f,g) and might not be distinguishable from 
# simply adding noise on (f,g) (which has the trivial impact of making the observations
# more noisy.  We can generate almost any pattern by tracing the x,y space in a
# certain manner.
# It seems we have the following order of flexibility re impact shape:
#   0. Deterministic (f,g)   
#   1. Noise on (f,g)   (f,g) ~ P
#   2. Random inputs    (x,y) ~ P
#   3. Confounding variable (x,y) = f(t) t ~ P
# 0 < 1 < 2 < 3 in terms of shape diversity
# 0: Deterministic looks very sharp and reproducible
# 1: Noise on (f,g) expands 0. into a diffuse volume of points
# 2: Random inputs create a volume of codom(f,g)
# 3: Confounding can create almost anything.
#
#
# We notice the following concrete Results:
# Case 1
# F,V,R = [L, x, x]: only one variable x, f,g linear in it 
# The relationship between f,g is linear; this is simple and expected as f = a*x
# and g = b * x translates to g = (b/a)*f.
# If we observe values for f,g and fit a model g = phi(f), phi =  linear would suffice.
# We therefore could map from f to g sing a simple model without knowing the distribution
# of x.
# Result: Everything deterministic and linear, no hidden vars -> Linearity conserved

# Case 2
# F,V,R = [L, x, (x,F)]: Like case 1 but with noise on the observations (f,g)
# The relationship between f,g is essentially linear but not very clearly visible.
# The clarity of the trend depends on noise level. One can clearly see (f,x), (g,x) 
# being linear in most cases (with noise reasonable) but a model f = phi(g) is
# harder to see. From f = a*x + n1; g = b*x + n2 we infer that the graph (f,g)
# has mean (a*x, b*x) - and then there is an isotropic jitter on top that blurs
# the line in both (= (f,g)) directions.
# Depending on noise level we could still diagnose that g = phi(f) is linear
# but it could be much more difficult. Also notice the blob-like appearance that
# does not lend itself well to fitting a line confidently; one would need to 
# account for the variation/randomness in both (f,g) via e.g. an error-in variables
# model. If we would only assume g to be noisily observed, then we would implicitly
# assume an incorrect stochastic model; a model for data that is distributed like
# in panel 2 rather than in panel 1 of the case 2 image.
# Result: Everything noisy and linear, no hidden vars -> Linearity partially 
#           buried under mulitvariate noise

# Case 3
# F,V,R = [L, (x,y) , (x(t), y(t))]: f = x, g = y but x,y are tied together
#               but purely deterministic
# The relationship between f,g, becomes nonlinear dependent on the relationship
# between x,y. The graph (f,g) shows the pattern of correlation between (x,y)
# more than it does the functional relationship f(x), g(y). That is on the one
# hand completely obvious but also means, we need to be careful when evaluationg
# relations between (f,g). Just diagnosing (f,g) we might be tempted to conclude
# that g = phi(f) where phi nonlinear. But as a matter of fact g does not depend 
# on f at all by construction; it is just the data pattern mistaken for causal
# relationsip. Imagining redoing the data gathering experiment on a subsequent 
# day under different conditions, then the (x,y) path would differ and the apparent
# nonlinear relation would be completely different. Only when sampling x,y separately
# would we find our graphs accurately describing that f doesnt depend on y and g 
# doesnt depend on x.
# Confoundingly, the dependence on a third underlying variable t means, that the
# graphs f(y) and g(x) look nontrivial as well. But this would not at all generalize
# to a different path x(t) y(t) in an independent 
# Result: Everything deterministic, hidden vars -> Nonlinearity introduced; depends
#            on sampling pattern


# Case 4
# F,V,R = [L, (x,y) , (x(t), y(t))]: f = a1*x + a2*y, g = b1*x + b2*y and x,y tied
#               but purely deterministic
# The difference to case 3 consists in f = f(x,y), g = g(x,y) so that both observed
# quantities depend on both input variables. What is visible then is the scatterplot
# pairs (f,x) - (g,x) and (f,y) - (g,y) look very similar but (g,f) is anyways a 
# nonlinear function. This goes to show that the nonlinearity does not only come
# from f,g being defined on a different set of input vars as is the case in case 3.
# Here f,g act very similar on x,y and nonetheless the f,g relationship is nonlinear.
# With more complicated x,y trajectories the nonlinearity becomes more pronounced.
# It is also entirely possible to create the impression of noisy observations by
# tuning the map from the confounding variable t. E.g. for x,y both nonlinear
# functions on t (above set nl_x = cos(15x), nl_y = cos(14y)), the impression is
# one of noise, the results look chaotically distributed (even though its a lissajous).
# In reality this scenario happens if exterior conditions parametrize the observations
# and these themselves are varying over e.g. time in a systematic manner.




# Case 5
# F,V,R = [L, (x,y) , (x(t), y(t), F)]: f = a1*x + a2*y, g = b1*x + b2*y and x,y tied,
#               and some noise on top.
# There is few surprises here. With noisy observations, the (f,g) graph can look
# like a noisy nonlinear relationship  or completely incoherent like (f,g) being
# independent. If the input variables x,y sweep the range of their possible values,
# the observed (f,g) graph might be indistinguishable from noise. Interestingly,
# the (f,x) (f,y), (g,x), (g,y) plots all look convinclingly like a noisly line
# for the cos / cos nonlinearities. For the id, arctan nonlinearlities, notice
# how heavily clustered and imbalanced the observed datasets can be.

    
# Case 6
# F,V,R = [L, (x,y) , (x,y)]: f = a1*x + a2*y, g = b1*x + b2*y; x,y random
#               but the maps are deterministic
# The difference to cases 3-5 is very visible. By not having x,y tied but sampled
# independently, we trace the x,y space randomly. No systematics means no clear
# pattern in the inputs and the output domain is sampled everywhere. No pretense
# towards a deterministic model can be made here. The results of this scenario
# with random inputs are indistinguishable through from the case 4 scenario with
# deterministic but (x,y) sweeping inputs. All the graphs but the (x,y) graph
# look very similar. 
# What is interesting to notice when compared to case 1 which was basically the 
# same case but with one input variable x instead of (x,y) is that (g,f) now 
# is not a simple function but the inclusion of another unobserved input variable
# has made (g,f) into a 2D domain. What was a simple line in case 1 has become
# diamond-shaped in case 6 simply because another unaccounted for variable was
# included.


    
# Case 7
# F,V,R = [L, (x,y) , (x,y)]: Like case 6 but with observations noisy.
# The only visible difference to case 6 is that (f,g) now looks less sharply defined.
# Whereas in case 6 we were tracing codom(f,g) which had some shard contours the 
# addition of noise waters this down. Interestingly, the (g,f) graph looks very 
# similar to the graph in case 2 where there is just one hidden input variable
# and noise on the outputs. It seems that the presence of hidden and unaccounted
# for variables would be hard to detect if the whole input space is sweeped and
# noise on top.
    

