import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
import pytest

from ddp_functions import crra
from ddp_functions import crra_prime
from ddp_functions import crra_dprime
from ddp_functions import f_dprime
from ddp_functions import f
from ddp_functions import consumption_ss
from ddp_functions import capital_ss

@pytest.fixture
def np_expected_utility():
    out = {}
    out["utility"] = np.array(
        [
            [
                [ 0.        ,  0.5       ,  0.66666667],
                [ 0.75      ,  0.8       ,  0.83333333],
                [ 0.85714286,  0.875     ,  0.88888889],
            ]
        ]
    )
    
    out["utility_prime"] = np.array(
        [
            [
                [1.        , 0.25      , 0.11111111],
                [0.0625    , 0.04      , 0.02777778],
                [0.02040816, 0.015625  , 0.01234568],
            ]
        ]
    )

    out["utility_dprime"] = np.array(
        [
            [
                [-2.        , -0.25      , -0.07407407],
                [-0.03125   , -0.016     , -0.00925926],
                [-0.0058309 , -0.00390625, -0.00274348],
            ]
        ]
    )
    return out

def test_float_utility():
    c = 10
    sigma = 2

    utility = crra(c, sigma)
    assert_equal(utility, 0.9 )

    utility_prime= crra_prime(c,sigma)
    assert_equal(utility_prime, 0.01 )

    utility_dprime = crra_dprime(c,sigma)
    assert_equal(utility_dprime, -0.002 )

def test_np_utility(np_expected_utility):
    c = np.array(
        [ 
                [
                    [ 1 , 2 , 3 ],
                    [ 4 , 5 , 6 ],
                    [ 7 , 8 , 9 ]
                ]
    ]
    )

    sigma = 2

    np_utility = crra(c, sigma)
    assert_array_almost_equal(

        np_utility, 
        np_expected_utility["utility"], 
        decimal=4
    )

    np_utility_prime = crra_prime(c,sigma)
    assert_array_almost_equal(

        np_utility_prime, 
        np_expected_utility["utility_prime"], 
        decimal=4
    )

    np_utility_dprime = crra_dprime(c,sigma)
    assert_array_almost_equal(

        np_utility_dprime, 
        np_expected_utility["utility_dprime"], 
        decimal=4
    )

def test_production():
    k = 10
    alpha = 0.1

    prod = f(k, alpha)
    assert_almost_equal(prod, 1.258925412 )

    prod_dprime = f_dprime(k,alpha)
    assert_almost_equal(prod_dprime,-0.001133032)

def test_steady_state():
    alpha = 0.1
    beta = 0.95
    delta = 0.1

    kss = capital_ss(alpha, beta, delta)
    assert_almost_equal(kss, 0.62510168570)

    css = consumption_ss(kss, alpha, delta)
    assert_almost_equal(css, 0.8915924043)



