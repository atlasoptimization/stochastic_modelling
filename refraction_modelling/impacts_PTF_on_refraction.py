#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to simulate how the equations for the refractive index
of air give rise to nonlinear or noisy behavior for two refraction angles beta_1,
beta_2. With this, we want to demonstrate the limits of the boeckem equation 
presupposing a linear relation between beta_1, beta_2. 
We show that a (beta_1, beta_2) devolves into something nonlinear and noisy when
input variables covary or are ignored altogether. In this sequence of investigations,
we will incrementally introduce complexity to the model underlying Boeckem.
Complicated behavior might therefore follow from misspecified assumptions rather
than from true functional behavior or randomness.
We showcase multiple models:
    F = Functional relation is always Edlen equation (*)
    Var = Input variables can be x or x,y
    Ran = Which parts of the model are random, can be x, t with x(t),y(t)  x,y, and F
    (*) as given by eq 12,8,14 pp 160 - 161 of "An updated Edlen Equation for 
        the refractive index of air by KP Birch and MH Downs 1993 Metrologia 30 155"
These lead to multiple possible combinations we will investigate.

This is a refraction specific investigation analogous to the script "impact_ignored_
inputvariables.py" in stochastic modelling/data_analysis_investigations

WARNING: A lot of these analysis are back-of-the-envelope, so take them with a grain
of salt and check everything!

To analyse these relations, we will do the following:
    1. Imports and definitions
    2. Simulate cases 1, 2
    3. Simulate cases 3, 4
    4. Simulate cases 5, 6
    5. Plots and illustrations
The specific cases and what can be inferred from them are listed down below.
    
We use the following abbreviations: 
    F = Functional model for beta_1(T,P) beta_2(T, P)
    L = E the modified Edlen equation
    V = Variables ( either beta_1,beta_2, depend only on T or on (T,P))
    R = Random (Multiple possibilities: T random, T,P Random, F = Function values
                random, i.e. noisy, S random and T,P = NL(S) where S is a 1d path)
    NL = Nonlinear function
    
For wavelengths we will always choose lambda_1 = 500, lambda_2 = 1000 nm for
the sake of simplicity. Humidity is set to 0 in this script; see impacts_on_boeckem_2.py
for an investigation in the humid case.
We make several simplifying assumptions: 
    
    Refraction computation:
    General equation acc to Moritz eq 14a (Zur Geometrie der Refraktion)
        beta = (1/s) \int_0^s (1/n) (dn/dz) x dx
    We simplify this to beta = dn/dz (via len*(1/n)*dn/dz, len =1, n ~1)
    (we assume whatever is needed here and are aware that this is not a valid 
     approximation this is ok, though - we want to qualitatively illustrate the
     impact that ignored vars and randomness might have on the shape of the graph
     (beta_1, beta_2).
    
    Refractive index computation;
    General equation according to Edlen / Birch and Down, roughly :
        Q(lambda) ~ (1/10^8) *(8343 + 2406294 * [(130 - sigma^2)]^(-1))
                                    + 15999*[38.9 - sigma^2]^(-1)
        (n-1)_TP = [(p * Q(lambda))/ 96095] * [1 + 10^(-8) * p * (0.6 - 0.009 *t)]/(1+ 0,003 * t)
        (n-1)_TPF = (n-1)_TP - 10^(-10)*[(f) * (3.73 - 0.04 * (sigma^2))]
        where 
        Q = dispersion term
        (n-1)_TP = ref index of dry air
        (n-1)_TPF =  ref index of moist air
        sigma = is vacuum wavenumber expressed in (mu m)^(-1)
        p is pressure in pa
        t is temperature in deg C
        f is partial pressure of water vapor in pa

    We simplify this to 
        (n-1)_TPF = G(T,P,lambda) - 4* 10^(-10) * f
        G(T,P,lambda) = 10^(-5) * [p * Q(lambda)] * [1 + 0.6*10^(-8)*p]/[1 + 0.003*t]
        Q(lambda) = 10^(-8)*[8000 + 2 400 000 /[130 -(1/lambda)^2]]
        
        where lambda = 1/(wavelength in mu m)

We model:
    lambda_1 = 0.5 mu_m
    lambda_2 = 1 mu_m
    
    p_1 = 101325 pa (standard atmosphere)
    p_2 = p_1 + grad_p
    grad_p = case dependent
    
    t_1 = 15 deg C (standard atmosphere)
    t_1 = t_1 + grad_t
    grad_t = case dependent
    
    f_1 = 0 pa ( standard atmosphere)
    f_2 = f_1 + grad_f
    grad_f = case dependent
    

f = const = 0

Case 1
F,V,R = [L, T, T]    
Generative model maximally simple : 
    f = const
    p = const
    grad_t = random uniform +-5 deg c


   
Case 2
F,V,R = [L, T, (T,F)]    
Generative model with noise : f = alpha*x + eps_1, g = beta*x + eps_2, 
                                sample x, sample eps


Case 3
F,V,R = [L, (T,P) , (T(s), P(s))]
Generative model with x,y covarying : f = alpha_1*x + alpha_2*y,
                                      g = beta_1*x + beta_2*y, 
                                sample t -> (x(t), y(t))


Case 4
F,V,R = [L, (x,y) , (x(t), y(t), F)]
Generative model with x,y covarying and noise : 
                                      f = alpha_1*x + alpha_2*y + eps_1,
                                      g = beta_1*x + beta_2*y + eps_2, 
                                sample t -> (x(t), y(t)), sample eps

    
Case 5
F,V,R = [L, (x,y) , (x, y)]
Generative model with x,y random : 
                                      f = alpha_1*x + alpha_2*y ,
                                      g = beta_1*x + beta_2*y , 
                                sample (x, y)

    
Case 6
F,V,R = [L, (x,y) , (x,y F)]
Generative model with x,y random and noise : 
                                      f = alpha_1*x + alpha_2*y + eps_1,
                                      g = beta_1*x + beta_2*y + eps_2, 
                                sample (x,y)), sample eps

    
    
"""
    
"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_pts = 500
variation_t = [-5,5]
variation_p = [-1000,1000]
variation_f = [0, 1000]

sigma_noise = 1e-6
unit = [0,1]

# Edlen equations
Q_l = lambda l : (10**(-8))*(8000 + 2400000/(130 -(1/l)**2))
# l in mu m
G_tpl = lambda t,p,l: (10**(-5)) * (p * Q_l(l)) * (1 + 0.6*(10**(-8))*p)/(1 + 0.003*t)
# p in pa, t in deg C
refractive_index_tpfl = lambda t,p,f,l: 1+ G_tpl(t,p,l) - 4* (10**(-10)) * f
# t in deg C, p in pa, l in mu m, f in pa

ref_angle = lambda n0, n1 : n0-n1
# result in rad

#wavelengths in mu_m
l_1 = 0.5
l_2 = 1

# Nonlinearities
f_nl_t = lambda s: 5*s

f_nl_p = lambda s: 1000* np.arctan(4*np.pi*s)


"""
    2. Simulate cases 1, 2
"""


# i) Standard atmosphere
p_0 = 101325    # pa
t_0 = 15        # deg C
f_0 = 0         # pa

t_0_expanded = t_0 * np.ones(n_pts)
p_0_expanded = p_0 * np.ones(n_pts)
f_0_expanded = f_0 * np.ones(n_pts)

n0_l1 = refractive_index_tpfl(t_0, p_0, f_0, l_1)
n0_l2 = refractive_index_tpfl(t_0, p_0, f_0, l_2)

# i) Case 1
# F,V,R = [L, T, T]    
# Generative model maximally simple : beta_k = edlen(t,p,f,l_k)
#     f = const
#     p = const
#     grad_t = random uniform +-5 deg c

# meteoparams
grad_t_c1 = np.random.uniform(variation_t[0], variation_t[1], size = n_pts)
t_1_c1 = t_0_expanded + grad_t_c1

p_1_c1 = p_0_expanded
f_1_c1 = f_0_expanded

# refraction - beta1
n0_l1_c1 = n0_l1 * np.ones(n_pts)
n1_l1_c1 = np.array([refractive_index_tpfl(t_1_c1[k], 
                                           p_1_c1[k],
                                           f_1_c1[k],
                                           l_1)
            for k in range(n_pts)])
beta1_c1 =  n0_l1_c1 - n1_l1_c1

# refraction - beta2
n0_l2_c1 = n0_l2 * np.ones(n_pts)
n1_l2_c1 = np.array([refractive_index_tpfl(t_1_c1[k], 
                                           p_1_c1[k],
                                           f_1_c1[k],
                                           l_2)
            for k in range(n_pts)])
beta2_c1 =  n0_l2_c1 - n1_l2_c1




# ii) Case 2
# F,V,R = [L, T, T]    
# Generative model maximally simple : beta_k = edlen(t,p,f,l_k) + noise
#     f = const
#     p = const
#     grad_t = random uniform +-5 deg c
# noise sampled from gaussian

noise_beta1_c2 = np.random.normal(0, sigma_noise, size = n_pts)
noise_beta2_c2 = np.random.normal(0, sigma_noise, size = n_pts)

beta1_c2 = beta1_c1 + noise_beta1_c2
beta2_c2 = beta2_c1 + noise_beta2_c2



"""
    3. Simulate cases 3, 4
"""


# i) Case 3
# F,V,R = [L, (T,P) , (T(s), P(s))]
# Generative model with T,P covarying : beta_k = edlen(t(s),p(s), f, l_k),
#                                 sample s -> (T(s), P(s))

s_c3 = np.random.uniform(unit[0], unit[1], size = n_pts)


# meteoparams
grad_t_c3 = f_nl_t(s_c3)
t_1_c3 = t_0_expanded + grad_t_c3

grad_p_c3 = f_nl_p(s_c3)
p_1_c3 = p_0_expanded + grad_p_c3
f_1_c3 = f_0_expanded

# refraction - beta1
n0_l1_c3 = n0_l1 * np.ones(n_pts)
n1_l1_c3 = np.array([refractive_index_tpfl(t_1_c3[k], 
                                           p_1_c3[k],
                                           f_1_c3[k],
                                           l_1)
            for k in range(n_pts)])
beta1_c3 =  n0_l1_c3 - n1_l1_c3

# refraction - beta2
n0_l2_c3 = n0_l2 * np.ones(n_pts)
n1_l2_c3 = np.array([refractive_index_tpfl(t_1_c3[k], 
                                           p_1_c3[k],
                                           f_1_c3[k],
                                           l_2)
            for k in range(n_pts)])
beta2_c3 =  n0_l2_c3 - n1_l2_c3


# ii) Case 4
# F,V,R = [L, (T,P) , (T(s), P(s),F)]
# Generative model with T,P covarying : beta_k = edlen(t(s),p(s), f, l_k) + noise,
#                                 sample s -> (T(s), P(s)), sample noise

noise_beta1_c4 = np.random.normal(0, sigma_noise, size = n_pts)
noise_beta2_c4 = np.random.normal(0, sigma_noise, size = n_pts)

beta1_c4 = beta1_c3 + noise_beta1_c4
beta2_c4 = beta2_c3 + noise_beta2_c4



"""
    4. Simulate cases 5, 6
"""


# i) Case 5
# F,V,R = [L, (T,P) , (T, P)]
# Generative model with T,P random : 
#                                       beta_1 = Edlen(t,p,f,l_1)
#                                       beta_1 = Edlen(t,p,f,l_2)
#                                 sample (T, P)

# meteoparams
grad_t_c5 = np.random.uniform(variation_t[0], variation_t[1], size = n_pts)
t_1_c5 = t_0_expanded + grad_t_c1

grad_p_c5 = np.random.uniform(variation_p[0], variation_p[1], size = n_pts)
p_1_c5 = p_0_expanded + grad_p_c5
f_1_c5 = f_0_expanded

# refraction - beta1
n0_l1_c5 = n0_l1 * np.ones(n_pts)
n1_l1_c5 = np.array([refractive_index_tpfl(t_1_c5[k], 
                                           p_1_c5[k],
                                           f_1_c5[k],
                                           l_1)
            for k in range(n_pts)])
beta1_c5 =  n0_l1_c5 - n1_l1_c5

# refraction - beta2
n0_l2_c5 = n0_l2 * np.ones(n_pts)
n1_l2_c5 = np.array([refractive_index_tpfl(t_1_c5[k], 
                                           p_1_c5[k],
                                           f_1_c5[k],
                                           l_2)
            for k in range(n_pts)])
beta2_c5 =  n0_l2_c5 - n1_l2_c5


# ii) Case 6
# F,V,R = [L, (T,P) , (T, P,F)]
# Generative model with T,P,F random : 
#                                       beta_1 = Edlen(t,p,f,l_1) + noise
#                                       beta_1 = Edlen(t,p,f,l_2) + noise
#                                 sample (T, P), noise from gaussian
    
noise_beta1_c6 = np.random.normal(0, sigma_noise, size = n_pts)
noise_beta2_c6 = np.random.normal(0, sigma_noise, size = n_pts)

beta1_c6 = beta1_c5 + noise_beta1_c6
beta2_c6 = beta2_c5 + noise_beta2_c6



"""
    5. Plots and illustrations
"""


# i) Case 1 plot beta1, beta2, beta1,T, beta2,T

fig1, axs1 = plt.subplots(1, 3, dpi = 300, figsize = (15,5))
fig1.suptitle('Case 1: F,V,R = [L, T, T] \n \n beta_1, beta_2 from Edlen \n T sampled 15 deg C +-5 deg C')
axs1[0].set_title(' beta_2 vs beta_1')
axs1[0].scatter(beta1_c1, beta2_c1)
axs1[0].set_xlabel('beta_1')
axs1[0].set_ylabel('beta_2')

axs1[1].set_title(' beta1 vs grad T')
axs1[1].scatter(grad_t_c1, beta1_c1)
axs1[1].set_xlabel('Variable T')
axs1[1].set_ylabel('Refraction angle beta1')

axs1[2].set_title(' beta2 vs grad T')
axs1[2].scatter(grad_t_c1, beta1_c1)
axs1[2].set_xlabel('Variable T')
axs1[2].set_ylabel('Refraction angle beta2')

plt.tight_layout()
plt.show()


# ii) Case 2 plot beta1, beta2, beta1,T, beta2,T

fig2, axs2 = plt.subplots(1, 3, dpi = 300, figsize = (15,5))
fig2.suptitle('Case 2: F,V,R = [L, T, (T,F)] \n \n beta_1, beta_2 from Edlen(t,p,f,l)'
              ' + noise \n T sampled 15 deg C +-5 deg C \n noise sampled from gaussian')
axs2[0].set_title(' beta_2 vs beta_1')
axs2[0].scatter(beta1_c2, beta2_c2)
axs2[0].set_xlabel('beta_1')
axs2[0].set_ylabel('beta_2')

axs2[1].set_title(' beta1 vs grad T')
axs2[1].scatter(grad_t_c1, beta1_c2)
axs2[1].set_xlabel('Variable T')
axs2[1].set_ylabel('Refraction angle beta1')

axs2[2].set_title(' beta2 vs grad T')
axs2[2].scatter(grad_t_c1, beta1_c2)
axs2[2].set_xlabel('Variable T')
axs2[2].set_ylabel('Refraction angle beta2')

plt.tight_layout()
plt.show()



# iii) Case 3 plot beta1, beta2, beta1,T, beta2,T

fig3, axs3 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig3.suptitle('Case 3: F,V,R = [L, (T,P) , (T(s)),P(s))]\n \n \n \n beta_1, beta_2 from Edlen(t,p,f,l)'
              ' \n T = nl(s) \n P=nl(s) \n s ~ U[0,1]')
axs3[0,0].set_title(' beta_2 vs beta_1')
axs3[0,0].scatter(beta1_c3, beta2_c3)
axs3[0,0].set_xlabel('beta_1')
axs3[0,0].set_ylabel('beta_2')

axs3[0,1].set_title(' beta_1 vs T')
axs3[0,1].scatter(t_1_c3, beta1_c3)
axs3[0,1].set_xlabel('Variable T')
axs3[0,1].set_ylabel('beta_1')

axs3[0,2].set_title(' beta_2 vs T')
axs3[0,2].scatter(t_1_c3, beta2_c3)
axs3[0,2].set_xlabel('Variable T')
axs3[0,2].set_ylabel('beta_2')

axs3[1,0].set_title(' P vs T')
axs3[1,0].scatter(t_1_c3, p_1_c3)
axs3[1,0].set_xlabel('Variable T')
axs3[1,0].set_ylabel('Variable P')

axs3[1,1].set_title(' beta_1 vs P')
axs3[1,1].scatter(p_1_c3, beta1_c3)
axs3[1,1].set_xlabel('Variable P')
axs3[1,1].set_ylabel('beta_1')

axs3[1,2].set_title(' beta_2 vs P')
axs3[1,2].scatter(p_1_c3, beta2_c3)
axs3[1,2].set_xlabel('Variable P')
axs3[1,2].set_ylabel('beta_2')


plt.tight_layout()
plt.show()


# iv) Case 4 plot beta1, beta2, beta1,T, beta2,T

fig4, axs4 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig4.suptitle('Case 4: F,V,R = [L, (T,P) , (T(s)),P(s),F)]\n \n \n \n beta_1, beta_2 from Edlen(t,p,f,l)'
              ' \n T = nl(s) \n P=nl(s) \n s ~ U[0,1], \n noise sampled from Gaussian')
axs4[0,0].set_title(' beta_2 vs beta_1')
axs4[0,0].scatter(beta1_c4, beta2_c4)
axs4[0,0].set_xlabel('beta_1')
axs4[0,0].set_ylabel('beta_2')

axs4[0,1].set_title(' beta_1 vs T')
axs4[0,1].scatter(t_1_c3, beta1_c4)
axs4[0,1].set_xlabel('Variable T')
axs4[0,1].set_ylabel('beta_1')

axs4[0,2].set_title(' beta_2 vs T')
axs4[0,2].scatter(t_1_c3, beta2_c4)
axs4[0,2].set_xlabel('Variable T')
axs4[0,2].set_ylabel('beta_2')

axs4[1,0].set_title(' P vs T')
axs4[1,0].scatter(t_1_c3, p_1_c3)
axs4[1,0].set_xlabel('Variable T')
axs4[1,0].set_ylabel('Variable P')

axs4[1,1].set_title(' beta_1 vs P')
axs4[1,1].scatter(p_1_c3, beta1_c4)
axs4[1,1].set_xlabel('Variable P')
axs4[1,1].set_ylabel('beta_1')

axs4[1,2].set_title(' beta_2 vs P')
axs4[1,2].scatter(p_1_c3, beta2_c4)
axs4[1,2].set_xlabel('Variable P')
axs4[1,2].set_ylabel('beta_2')


plt.tight_layout()
plt.show()




# v) Case 5 plot beta1, beta2, beta1,T, beta2,T

fig5, axs5 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig5.suptitle('Case 5: F,V,R = [L, (T,P) , (T,P)]\n \n \n \n beta_1, beta_2 from Edlen(t,p,f,l)'
              ' \n T sampled 15 deg C +-5 deg C \n P sampled 1013 hpa +- 10 hpa')
axs5[0,0].set_title(' beta_2 vs beta_1')
axs5[0,0].scatter(beta1_c5, beta2_c5)
axs5[0,0].set_xlabel('beta_1')
axs5[0,0].set_ylabel('beta_2')

axs5[0,1].set_title(' beta_1 vs T')
axs5[0,1].scatter(t_1_c5, beta1_c5)
axs5[0,1].set_xlabel('Variable T')
axs5[0,1].set_ylabel('beta_1')

axs5[0,2].set_title(' beta_2 vs T')
axs5[0,2].scatter(t_1_c5, beta2_c5)
axs5[0,2].set_xlabel('Variable T')
axs5[0,2].set_ylabel('beta_2')

axs5[1,0].set_title(' P vs T')
axs5[1,0].scatter(t_1_c5, p_1_c5)
axs5[1,0].set_xlabel('Variable T')
axs5[1,0].set_ylabel('Variable P')

axs5[1,1].set_title(' beta_1 vs P')
axs5[1,1].scatter(p_1_c5, beta1_c5)
axs5[1,1].set_xlabel('Variable P')
axs5[1,1].set_ylabel('beta_1')

axs5[1,2].set_title(' beta_2 vs P')
axs5[1,2].scatter(p_1_c5, beta2_c5)
axs5[1,2].set_xlabel('Variable P')
axs5[1,2].set_ylabel('beta_2')


plt.tight_layout()
plt.show()



# vi) Case 6 plot beta1, beta2, beta1,T, beta2,T

fig6, axs6 = plt.subplots(2, 3, dpi = 300, figsize = (15,10))
fig6.suptitle('Case 6: F,V,R = [L, (T,P) , (T,P,F)]\n \n \n \n beta_1, beta_2 from Edlen(t,p,f,l) + noise'
              ' \n T sampled 15 deg C +-5 deg C \n P sampled 1013 hpa +- 10 hpa \n Noise from gaussian')
axs6[0,0].set_title(' beta_2 vs beta_1')
axs6[0,0].scatter(beta1_c6, beta2_c6)
axs6[0,0].set_xlabel('beta_1')
axs6[0,0].set_ylabel('beta_2')

axs6[0,1].set_title(' beta_1 vs T')
axs6[0,1].scatter(t_1_c5, beta1_c6)
axs6[0,1].set_xlabel('Variable T')
axs6[0,1].set_ylabel('beta_1')

axs6[0,2].set_title(' beta_2 vs T')
axs6[0,2].scatter(t_1_c5, beta2_c6)
axs6[0,2].set_xlabel('Variable T')
axs6[0,2].set_ylabel('beta_2')

axs6[1,0].set_title(' P vs T')
axs6[1,0].scatter(t_1_c5, p_1_c5)
axs6[1,0].set_xlabel('Variable T')
axs6[1,0].set_ylabel('Variable P')

axs6[1,1].set_title(' beta_1 vs P')
axs6[1,1].scatter(p_1_c5, beta1_c6)
axs6[1,1].set_xlabel('Variable P')
axs6[1,1].set_ylabel('beta_1')

axs6[1,2].set_title(' beta_2 vs P')
axs6[1,2].scatter(p_1_c5, beta2_c6)
axs6[1,2].set_xlabel('Variable P')
axs6[1,2].set_ylabel('beta_2')


plt.tight_layout()
plt.show()

