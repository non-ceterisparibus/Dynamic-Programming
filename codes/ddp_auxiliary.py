
import numpy as np
from numba import jit

from codes.ddp_functions import crra
from codes.ddp_functions import capital_ss
from codes.ddp_functions import linearized_dynamic_system


@jit
def consum_grid_mat(k_grid,alpha,delta):
    """

    """
    k_state = np.matrix(k_grid)
    k_action = np.transpose(k_state)
    c = np.power(k_state, alpha) + (1 - delta)*k_state - k_action
    c_grid = np.asarray(c)
    return c_grid


def get_grid(dev, num_states, beta, alpha, delta, sigma):
    """
    Get current-period utility payoff
    implied by moving from some K_state (today) to some K_action (next period)
   
    """
    # Capitals at steady state
    capitals = capital_ss(alpha, beta, delta)
    # Setting defaul deviation from steady state
    if dev is None:
        dev = 0.2  
    grid_min, grid_max = (1.0 - dev)*capitals, (1.0 + dev)*capitals

    # Capital grid
    k_grid = np.linspace(grid_min, grid_max, num_states)
    
    # Consumption grid
    c_grid = consum_grid_mat(k_grid,alpha,delta)
    c_grid[c_grid < 0] = np.nan

    # Utility grid
    u_grid = crra(c_grid, sigma)

    return k_grid, c_grid, u_grid


def _consumption_linear_solution(k_grid, alpha, beta, delta, sigma):
    """
    """
    capitals, consumptions, A = linearized_dynamic_system(alpha, beta, delta, sigma)

    w, v = np.linalg.eig(A)
    stab_col_ind = np.where(w < 1)

    stab_ind = (v[0,stab_col_ind]/v[1,stab_col_ind])
    C_lin = stab_ind[0][0]*(k_grid - capitals) + consumptions

    return C_lin

def linear_solution(num_states, k_grid , T, alpha, beta, delta, sigma):
    """
    Randomly choose a policy from plausible choices from grid world


    Parameters
    ----------
    T : time vector, of length n
        
    """
    capitals, consumptions, A = linearized_dynamic_system(alpha, beta, delta, sigma)

    w, v = np.linalg.eig(A)
    stab_col_ind = np.where(w < 1)

    stab_ind = (v[0,stab_col_ind]/v[1,stab_col_ind])

    k_sim = np.tile(np.nan, [num_states, T])        # capitals simulated
    c_sim = np.tile(np.nan, [num_states, T])        # consumptions simulated
    x_t = np.tile(np.nan, [2, T])                   # deviation of capital and consumption

    for i, k in enumerate(k_grid):

        x_t[1,0] = k - capitals                     #capitals deviation
        x_t[0,0] = stab_ind[0][0]*x_t[1,0]          #consumptions deviation

        for t in range(T-1):
            x_t[1,t+1] = np.matmul(A[1,:],x_t[:,t])
            x_t[0,t+1] = stab_ind[0][0]*x_t[1,t+1]
        
        # Computing levels
        k_sim[i,:] = x_t[1,:] + capitals
        c_sim[i,:] = x_t[0,:] + consumptions

    return k_sim, c_sim

