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
    3. Simulate cases 3, 4
    4. Simulate cases 5, 6
    5. Plots and illustrations
The specific cases and what can be inferred from them are listed down below.
    
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
Generative model with x,y covarying : f = alpha_1*x + alpha_2*y,
                                      g = beta_1*x + beta_2*y, 
                                sample t -> (x(t), y(t))
    Then since there is a single random variable t determining a path x(t), y(t),
    x,y always vary together in a nonlinear way leading to nonlinearity in f,g
    This situation is akin to x,y both depending on an unaccounted for third
    factor. They individually vary and their variation is coupled.

Case 4
F,V,R = [L, (x,y) , (x(t), y(t), F)]
Generative model with x,y covarying and noise : 
                                      f = alpha_1*x + alpha_2*y + eps_1,
                                      g = beta_1*x + beta_2*y + eps_2, 
                                sample t -> (x(t), y(t)), sample eps
    This is very similar to case 3 in terms of construction but with noise added
    on top of the observations. The result is f,g varying nonlinearly and noisily 
    with the underlying pattern hard to distinguish.
    
Case 5
F,V,R = [L, (x,y) , (x, y)]
Generative model with x,y random : 
                                      f = alpha_1*x + alpha_2*y ,
                                      g = beta_1*x + beta_2*y , 
                                sample (x, y)
    This looks insterestingly different from the case where the variation in x,y
    is explainable by a single variable introducing a nonlinear pattern. The results
    simply look more disordered with less structure visible
    
Case 6
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
sigma_noise = 0.01

alpha = 1
beta = 1

alpha_1 = 1
alpha_2 = 1
beta_1 = 2
beta_2 = 0.5



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
    3. Simulate cases 3, 4
"""


# i) Case 3
# F,V,R = [L, (x,y) , (x(t), y(t))]
# Generative model with x,y covarying : f = alpha_1*x + alpha_2*y,
#                                       g = beta_1*x + beta_2*y, 
#                                 sample t -> (x(t), y(t))
#     Then since there is a single random variable t determining a path x(t), y(t),
#     x,y always vary together in a nonlinear way leading to nonlinearity in f,g
#     This situation is akin to x,y both depending on an unaccounted for third
#     factor. They individually vary and their variation is coupled.

t_c3 = np.random.uniform(unit[0], unit[1], size = n_pts)

x_c3 = t_c3
y_c3 = np.cos(2*np.pi*t_c3)

f_c3 = alpha_1*x_c3 + alpha_2*y_c3
g_c3 = beta_1*x_c3 + beta_2*y_c3


# ii) Case 4
# F,V,R = [L, (x,y) , (x(t), y(t), F)]
# Generative model with x,y covarying and noise : 
#                                       f = alpha_1*x + alpha_2*y + eps_1,
#                                       g = beta_1*x + beta_2*y + eps_2, 
#                                 sample t -> (x(t), y(t)), sample eps
#     This is very similar to case 3 in terms of construction but with noise added
#     on top of the observations. The result is f,g varying nonlinearly and noisily 
#     with the underlying pattern hard to distinguish.

t_c4 = np.random.uniform(unit[0], unit[1], size = n_pts)

x_c4 = t_c4
y_c4 = np.cos(2*np.pi*t_c3)

noise_f_c4 = np.random.normal(0, sigma_noise, size = n_pts)
noise_g_c4 = np.random.normal(0, sigma_noise, size = n_pts)

f_c4 = alpha_1*x_c4 + alpha_2*y_c4 + noise_f_c4
g_c4 = beta_1*x_c4 + beta_2*y_c4 + noise_g_c4


"""
    4. Simulate cases 5, 6
"""


# i) Case 5
# F,V,R = [L, (x,y) , (x, y)]
# Generative model with x,y random : 
#                                       f = alpha_1*x + alpha_2*y ,
#                                       g = beta_1*x + beta_2*y , 
#                                 sample (x, y)
#     This looks insterestingly different from the case where the variation in x,y
#     is explainable by a single variable introducing a nonlinear pattern. The results
#     simply look more disordered with less structure visible


x_c5 = np.random.normal(0, sigma_noise, size = n_pts)
y_c5 = np.random.normal(0, sigma_noise, size = n_pts)

f_c5 = alpha_1*x_c5 + alpha_2*y_c5
g_c5 = beta_1*x_c5 + beta_2*y_c5


# ii) Case 6
# F,V,R = [L, (x,y) , (x,y F)]
# Generative model with x,y random and noise : 
#                                       f = alpha_1*x + alpha_2*y + eps_1,
#                                       g = beta_1*x + beta_2*y + eps_2, 
#                                 sample (x,y)), sample eps
#     No qualititative difference in visualization compared to case 5. The relationships
#     are lost to noise even more heavily.
    
x_c6 = np.random.normal(0, sigma_noise, size = n_pts)
y_c6 = np.random.normal(0, sigma_noise, size = n_pts)

noise_f_c6 = np.random.normal(0, sigma_noise, size = n_pts)
noise_g_c6 = np.random.normal(0, sigma_noise, size = n_pts)

f_c6 = alpha_1*x_c6 + alpha_2*y_c6 + noise_f_c6
g_c6 = beta_1*x_c6 + beta_2*y_c6 + noise_g_c6



"""
    5. Plots and illustrations
"""


# i) Case 1 plot f,g ; f,x ; g,x

fig1, axs1 = plt.subplots(1, 3, dpi = 300, figsize = (15,5))
fig1.suptitle('Case 1: F,V,R = [L, x, x]')
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
fig2.suptitle('Case 2: F,V,R = [L, x, (x,F)]')
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

fig3, axs3 = plt.subplots(2, 3, dpi = 300, figsize = (15,5))
fig3.suptitle('Case 3: F,V,R = [L, (x,y), (x(t), y(t))]')
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

fig4, axs4 = plt.subplots(2, 3, dpi = 300, figsize = (15,5))
fig4.suptitle('Case 4: F,V,R = [L, (x,y), (x(t), y(t), F)]')
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

fig5, axs5 = plt.subplots(2, 3, dpi = 300, figsize = (15,5))
fig5.suptitle('Case 4: F,V,R = [L, (x,y), (x,y)]')
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

fig6, axs6 = plt.subplots(2, 3, dpi = 300, figsize = (15,5))
fig6.suptitle('Case 4: F,V,R = [L, (x,y), (x,y)]')
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




