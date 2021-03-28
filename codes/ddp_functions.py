"""

"""
import numpy as np

def f(k, alpha):
    """
    Cobb-Douglas production function
    """
    return k**alpha

def f_dprime(k, alpha):
    """
    Second derivative of the production function
    """
    return alpha *(alpha - 1)* k**(alpha - 2)

def crra_prime(c,sigma):
    """
    First derivative of the utility function
    """
    return 1/(c**(sigma))

def crra_dprime(c,sigma):
    """
    First derivative of the utility function
    """
    return -sigma/(c**(1+sigma))

def crra(c, sigma):
    """
    Return the CRRA utility evaluation of *argument* where the Arrow-Pratt
    coefficient of relative risk aversion takes on the value *exponent*.
    Neither handle cases where *argument* is less than nor log utility.
    Utility = (C^(1 - sigma) - 1)/(1 - sigma)
    """

    numerator = c ** (1.0 - sigma)
    denominator = 1.0 - sigma
    utility = np.divide(numerator - 1, denominator)

    return utility

def capital_ss(alpha, beta, delta):
    """
    Calculate the capital at the steady state for:
    production function f = k^alpha 
    and utility function    U = (c^(1 - sigma) - 1)/(1 - sigma)
    """

    return ( (1-beta+delta*beta) / (alpha * beta) )**(1/(alpha-1))

def consumption_ss(capitals, alpha, delta):
    """
    Calculate the consumption at the steady state for:
    production function f = k^alpha 
    and utility function U = (c^(1 - sigma) - 1)/(1 - sigma)
    """

    return capitals**alpha - delta*capitals

def linearized_dynamic_system(alpha, beta, delta, sigma):
    """
    the linearly approximated solution
    just in case we later want to compare it to the non-linear policy functions obtained from dynamic programming
    For the linearly approximated solution,
    we need to get entries of the A matrix, using steady state values and corresponding derivatives.

    """
    capitals = capital_ss(alpha, beta, delta)
    consumptions = consumption_ss(capitals, alpha, delta)

    coeff = crra_prime(consumptions,sigma)*f_dprime(capitals, alpha)/crra_dprime(consumptions,sigma)

    A = np.zeros((2, 2))
    A[0][0] = 1.0 + beta*coeff
    A[0][1] = -coeff
    A[1][0] = -1
    A[1][1] = 1.0/beta

    return capitals, consumptions, A


