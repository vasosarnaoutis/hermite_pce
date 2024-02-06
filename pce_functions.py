'''
Basic polynomial chaos implementation for hermite polynomials implemented by V. Arnaoutis
For any mistakes please contact the creator
This was implemented and tested in python3.8, compatibility with earlier versions is not guaranteed
'''

import numpy as np
import math
import numpoly
from sklearn.linear_model import LinearRegression
from scipy.special import factorial

### Flat Expansion Generator #####
def create_poly_by_coefficient(coef,q):
    expansion = []
    for i , c_i in enumerate(coef):
        expansion.append(c_i*q**i)
    return chaospy.polynomial(expansion)

### Hermite Polynomial Generator ###
# With variable q and degree n
def hermite_poly(q_i,n):
    """
    Used internally in the class to compute the hermite polynomial of a specific order
    :param q_i:
    :param n:
    :return: hermite polynomial of R.V. q_i and order n
    """
    if (n ==0):
        return 1.0
    if (n == 1):
        return q_i
    h = []
    h.append(1.0)
    h.append(q_i)
    for n_i in range(2,n+1):
        h.append(q_i*h[n_i-1] - (n_i-1)*h[n_i-2])
    return h[-1]

### Generate polynomial expansion using a basis matrix of multi_index
### And variables q
def generate_expansion(basis, q):
    """
    Generate a hermite polynomial expansion using the basis array provided and the random variables q
    :param basis:
    :param q:
    :return: a numpoly polynomial expansion
    """
    poly_basis_expanded = []
    for degree in range(len(basis)):
        h_degree = []
        if q.size == 1:
            h_degree = [hermite_poly(q,basis[degree][0])]
        else:
            for var_i in range(q.size):
                h_degree.append(hermite_poly(q[var_i],basis[degree][var_i]))
        poly_basis_expanded.append(math.prod(h_degree))
    poly_expanded = numpoly.polynomial(poly_basis_expanded)
    return poly_expanded.reshape(-1,1)


def get_fitted_model(poly_expansion, samples, sample_evals):
    """
    Applied linear regression on the polynomial expansion provided based on samples and their evaluation
    :param poly_expansion:
    :param samples:
    :param sample_evals:
    :return: the fitted model, used to collect the .coef_
    """
    ## Sampling and fitting the coefficients manually
    poly_sampled = np.squeeze(poly_expansion(*samples)) #squeeze maintains a 2-D array neccessary for .fit()
    custom_model = LinearRegression(fit_intercept=False, n_jobs=-1)  # n_jobs = -1 for parallel computing
    custom_model.fit(poly_sampled.T, sample_evals)
    #custom_coef = custom_model.coef_
    return custom_model


def adjust_order(coefficient, largest_coeff):
    """
    Add zeros to the list of coefficients to match the size of the largest list of coefficients
    :param coefficient: coefficients of interest, to be adjusted
    :param largest_coeff: coefficient list of the target size
    :return: modified coefficient to equal the shape of largest_coeff
    """
    size_out, size_coef = coefficient.shape
    max_size_out,max_size_coef = largest_coeff.shape
    return np.pad(coefficient,[(0,0),(0,max_size_coef-size_coef)],'constant')

def sample(coefficient, expansion, sample):
    """
    Evaluates the samples of the RV based on a poly_expansion and its coefficients
    :param coefficient:
    :param expansion:
    :param sample:
    :return: Returns the outcome of the coeff*exp_basis(samples)
    Note: This works for a single vector of coefficients, for matrix try:
    np.squeeze(np.matmul(coeff,expansion)(*samples))
    """
    return (coefficient*expansion)(*sample).sum(axis=0)

def cov_basis(basis,q):
    """
    Calculates the cov of the basis for a set of R.V
    using the inner product of the factorial of the basis
    :param basis: The matrix basis (Hermite)
    :param q: the RV list
    :return: cov of basis
    """
    if q.size == 1:
        mvar_inner = [factorial(basis[i][0]) for i in range(len(basis))]
    else:
        ## compute inner produc, for scalars inner product of a,b,c is a*b*c
        ## Used to be np.inner(*factorial) but substituted for higher than 2 RV
        mvar_inner = [np.prod(factorial(basis[i],exact=True)) for i in range(len(basis))]

    mvar_inner[0] = 0  # for scalar
    cov_mvar = np.diag(mvar_inner)
    return cov_mvar


def E(coeff, expansion):
    """
    The expected value of a distribution or polynomial.

    1st order statistics of a probability distribution or polynomial on a
    Hermite probability space with R.V. of normal distribution with s.t.d. 1 and mean 0.

    A counter-intuitive approach to do this. This could simply be coeff[0]
    """
    return (coeff*expansion)(*np.zeros(len(expansion.indeterminants))).sum(axis=-2)
