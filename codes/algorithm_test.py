import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from ddp_algorithms import evaluate_policy
from ddp_algorithms import bellman_equation
from ddp_algorithms import state_wise_max

@pytest.fixture
def setup_value_bellman():
    out = {}
    out["u_grid"] = np.array(
        [
            [
                [ 0.        ,  0.5       ,  0.66666667],
                [ 0.75      ,  0.8       ,  0.83333333],
                [ 0.85714286,  0.875     ,  0.88888889],
            ]
        ]

    )

    out["val_old"] = np.array(
        [
            [
                [ 1.        ,  1.        ,  1.],
                [ 1.        ,  1.        ,  1.],
                [ 1.        ,  1.        ,  1.],
            ]
        ]
    )

    out["beta"] = 0.95
    return out

@pytest.fixture
def setup_state_wise_max():
    out = np.array(
        [
            [ 0.        ,  0.5       ,  0.66666667],
            [ 0.75      ,  0.8       ,  0.83333333],
            [ 0.85714286,  0.875     ,  0.88888889]
        ]
    )
    return out

@pytest.fixture
def setup_evaluation_policy():
    out = {}
    out["policy"] = np.array([1, 1, 1])
    out["u_grid"] = np.array(
            [
                [ 0.        ,  0.5       ,  0.66666667],
                [ 0.75      ,  0.8       ,  0.83333333],
                [ 0.85714286,  0.875     ,  0.88888889],
            ]
    )
    out["beta"] = 0.95
    return out

@pytest.fixture
def expected_value_bellman():
    out= np.array(
        [
            [
            [
                [0.95      , 1.45      , 1.61666667],
                [1.7       , 1.75      , 1.78333333],
                [1.80714286, 1.825     , 1.83888889],
            ]
            ]
        ]
    )
    return out

@pytest.fixture
def expected_state_wise_max():
    out = {}
    out["value_function"] = np.array([0.85714286,  0.875  ,  0.88888889])
    out["policy"] = np.array([2 , 2 , 2])
    return out

@pytest.fixture
def expected_evaluate_policy():

    out = np.array([15.95    , 16.     , 16.03333333])
    
    return out

def test_bellman_equation(setup_value_bellman, expected_value_bellman):
    v_bellman = bellman_equation(**setup_value_bellman)
    assert_array_almost_equal(

        v_bellman,
        expected_value_bellman, 
        decimal=6
    )

def test_state_wise_max(setup_state_wise_max, expected_state_wise_max):
    value_max, policy = state_wise_max(setup_state_wise_max)
    assert_array_almost_equal(

        value_max, 
        expected_state_wise_max["value_function"], 
        decimal=6
    )

    assert_array_equal(

        policy, 
        expected_state_wise_max["policy"]
    )

def test_evaluate_policy(setup_evaluation_policy,expected_evaluate_policy):
    value_policy = evaluate_policy(**setup_evaluation_policy)
    assert_array_almost_equal(
        value_policy, 
        expected_evaluate_policy, 
        decimal=6
    )