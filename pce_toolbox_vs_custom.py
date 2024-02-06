'''
Validation of pce_functions script with chaospy library implemented by V. Arnaoutis
This requires the original pce_functions.py script to run with
'''

### Compare the chaospy toolbox with custom made pce functions
### For n variable system of Normal Distribution and Hermite Polynomials
### Hermite polynomials
# H0 = 1
# H1 = x
# H2 = x2 - 1
# H3 = x3 -3x
# H4 = x4 - 6x2 +3
# H5 = x5 - 10x3 +15x
# H6 = x6 - 15x4 + 45x2 - 15
# H7 = x7 - 21x5 + 105x3 - 105x

import chaospy
from matplotlib import pyplot as plt
import numpy as np
import math
import numpoly
from sklearn.linear_model import LinearRegression
import time
from pce_functions import *


#####################################
##### BUILD FROM CHAOSPY LIBRAR #####
#####################################

##### GENERATE SAMPLES
#q0 = chaospy.variable(1)
n_variables = 2
#mass_q =  chaospy.Uniform(0, 1)
th = [chaospy.Normal(0,1) for i in range(n_variables)]
th_joint = chaospy.J(*th)

##### SELECT POLYNOMIAL EXPANSION
th0_expansions = [chaospy.generate_expansion(order, th[0])
                for order in range(1, 10)]

th_joint_expansions = [chaospy.generate_expansion(order, th_joint)
                for order in range(1, 10)]


#####################################
########### BUILD CUSTOM ############
#####################################

## Generate random variables and the hermite basis for two variables
# Example of basis - lexicographical multi-indices of monomial exponents
# hermite_basis = [[0 ,0],
#                  [1, 0],
#                  [0 ,1],
#                  [2 ,0],
#                  [1, 1],
#                  [0, 2]]

## Example usage of hermite_poly
# How to call specific Hermite polynomial degree
#h2= hermite_poly(q[0],2)

q = numpoly.variable(n_variables)
order = 2
hermite_basis = numpoly.glexindex(order+1,dimensions=n_variables, graded= True)
poly_expansion = generate_expansion(hermite_basis,q)


#### SAMPLING FROM MODEL
samples = th_joint.sample(10000)
#samples_custom = np.random.normal(0,1,(2,10000)) # manual sampling

## keep in mind this is the stupidest model, its basically y(omega1,omega2)=10, so its actually deterministic
# (all coefficients are expected to be 0 except the first one)
# You should create some function y = f(samples) for more "advanced" problems
evals = np.ones(10000)*10 #should create some function y for more advanced problems
# an exmaple would be :
# evals = 5+ samples[0] + 2*samples[1]
# which is basically y = 5 + N(0,1) +2N(0,1) where N are two i.i.d. RVs
# so one would expect the mean to be 5 (both N are zero-centered) and 1st order coefficients to be [1 2]

##### COEFFICIENT FITTING FOR REGRESSION
## note that for th_joint_expansions order starts from 1, so [order-1] just corrects dimensions for python lists
chaos_model =chaospy.fit_regression(th_joint_expansions[order-1], samples, evals)
chaos_coef = chaos_model.coefficients

## Sampling and fitting the coefficients manually
custom_model = get_fitted_model(poly_expansion,  samples, evals)
custom_coef = custom_model.coef_


## Compute Expectation for hermite polynomial (normal distribution)
tool_e = chaospy.E(chaos_model,th_joint)
custom_e_val = chaospy.E(np.sum(custom_coef*poly_expansion),th_joint)
custom_e = custom_coef[0]

## Compute covariance for hermite polynomial (only custom)
# naming might be misdirecting here, what cov_basis actually compute its the
# \Delta which is a diagonal matrix of factorial a! and first entry 0
# so that cov(x,x)=cov(x) = x_a Delta x_a, with x_a the coefficients
delta = cov_basis(hermite_basis,q)
cov_custom = np.matmul(np.matmul(custom_coef, delta), custom_coef) #one would need to transpose in case of multi-state


print("Done")


## compact form to be reused in other scripts (only for custom toolbox)
'''
import numpoly
import pce_functions as pce

## Initiate PCE ##
n_rvs = 2
# SELECT POLYNOMIAL EXPANSION
order = 2 # 0:mean 1:+st.d. 2:+skew
q_c = numpoly.variable(n_variables)
hermite_basis = numpoly.glexindex(order+1,dimensions=n_variables, graded=True)
poly_expansion = pce.generate_expansion(hermite_basis,q_c)
poly_expansion_coeff = poly_expansion.sum().coefficients
cov_mvar = pce.cov_basis(hermite_basis,q_c)
'''
