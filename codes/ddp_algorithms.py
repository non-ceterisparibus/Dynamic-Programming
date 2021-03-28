"""Solution algorithms"""

import numpy as np

def bellman_equation(u_grid, val_old, beta):
    """
    Computes and returns the updated
    value function for an old value function `val_old`.
    Return NA position to 0 value

    Parameters
    ----------
    val_old : array_like(float, ndim=1)
        Old value function vector, of length n.

    u_grid : array_like( 2-dimensional ndarray of shape (n, n))
        Utility grid.

    Returns
    -------
    v_new : array_like( 2-dimensional ndarray of shape (n, n))
        Updated utility value grid

    """
    v = u_grid + beta*val_old[:,None]

    v_new = np.where(np.isnan(u_grid), 0, v)

    return v_new

def state_wise_max(u_grid):
    """
    Find the maximum value function and the corresponding policy
    
    Parameters
    ----------
    u_grid : array_like( 2-dimensional ndarray of shape (n, n))
            Utility grid

    Returns
    -------
    value_fn : array_like(float, ndim=1)
            Value function vector, of length n.

    policy : ndarray(float, ndim=2)
        Transition matrix for `policy`, of shape (n, n).
    """
    u_new = np.where(np.isnan(u_grid), 0, u_grid)

    value_fn = np.amax(u_new, axis = 0)
    policy = np.argmax(u_new, axis = 0)
    return value_fn, policy

def evaluate_policy(policy, u_grid, beta):
    """
    Computes the updated value function `Tv` for a `policy`.

    Parameters
    ----------
    policy : array_like(int, ndim=1)
        Policy vector, of length n.
        
    u_grid : array_like( 2-dimensional ndarray of shape (n, n))
            Utility grid

    U_policy : array_like(float, ndim=1)
            Utility vector corresponding with policy, of length n
    
    Q_policy : array_like( 2-dimensional ndarray of shape (n, n))
             sparse matrix Q with zeros everywhere except for its row i and column j elements

    Returns
    -------
    v_policy : array_like(float, ndim=1)
        Value function vector, of length n

    """
    num_states = len(u_grid)

    # Solve (I - beta * Q_policy) v = U_policy
    U_policy = u_grid[policy,range(len(policy))]
    b = U_policy

    Q_policy = np.zeros((num_states, num_states))
    Q_policy[range(len(policy)),policy] = 1.0
    
    I = np.identity(num_states)
    A = I - beta * Q_policy

    v_policy = np.linalg.solve(A, b)
    return v_policy

def v_greedy(v, u_grid, beta):
    """
    Parameters
    ----------
    v : array_like(float, ndim=1)
        Value function vector, of length n.

    policy : ndarray(int, ndim=1)
        Optional output array for `sigma`.
    Returns
    -------
    policy : ndarray(int, ndim=1)
        v-greedy policy vector, of length n.
    """

    #improve value function
    v_new = bellman_equation(u_grid, v, beta)
    
    #find the new_policy (new maximum)
    new_value , new_policy = state_wise_max(v_new)
    return new_value, new_policy

def random_policy(_grid):
    """
    Randomly choose a policy from plausible choices from grid world

    Parameters
    ----------
    c_grid : array_like( 2-dimensional ndarray of shape (n, n))
        Utility vector, of length n

    Returns
    -------
    policy : ndarray(int, ndim=1)
            random policy vector, of length n
    """
    new_grid = np.where(np.isnan(_grid), 0, _grid)

    a = np.transpose(new_grid)

    policy = [np.random.choice(r.nonzero()[0]) for r in a]

    return policy

def value_iteration(crit, max_iter, u_grid, beta):
    """
    Solve the optimization problem by value iteration.

    """
    if crit is None:
        crit = 1e-06
    
    num_states = len(u_grid)

    # set up
    value_store_iter = np.tile(np.nan, [max_iter, num_states])
    store_policy = np.zeros((max_iter, num_states), dtype= int)
    val_old, val_new = np.zeros(num_states), np.zeros(num_states)

    for i in range(max_iter):
        v_new = bellman_equation(u_grid, val_old, beta)
        val_new, store_policy[i,:] = state_wise_max(v_new)
        
        max_diff = np.max(np.absolute(val_new - val_old))
        # quit iterations, when convergence is achieved
        if max_diff < crit:
            break

        value_store_iter[i,:] = val_new
        val_old = val_new

    num_iter = i

    return value_store_iter, store_policy, num_iter

def policy_iteration(max_iter, u_grid, beta):
    """
    Solve the optimization problem by policy iteration

    matrix Q with zeros everywhere except for its row i and column j elements, which equal one

    """

    num_states = len(u_grid)

    # set up
    value_store_iter = np.tile(np.nan, [max_iter, num_states])
    store_policy = np.tile(np.nan, [max_iter, num_states])

    # Initialize with a random policy and initial value function
    policy = random_policy(u_grid)
    v_policy = evaluate_policy(policy, u_grid, beta)

    for i in range(max_iter):
        # Policy improvement
        improved_value , improved_policy = v_greedy(v_policy, u_grid, beta)
    
        # Policy evaluation
        Tv = evaluate_policy(improved_policy, u_grid, beta)

        store_policy[i+1,:] = improved_policy
        value_store_iter[i+1,:] = improved_value

        # quit iterations, when convergence is achieved
        if np.array_equal(improved_policy, policy):
            break

        policy = improved_policy
        v_policy = Tv

    num_iter = i + 1

    return value_store_iter, store_policy, num_iter

def modified_policy_iteration(crit, k, max_iter, u_grid, beta, ):
    """
    Solve the optimization problem by policy iteration

    """
    if k is None:
        k = 30
    
    if crit is None:
        crit = 1e-06

    num_states = len(u_grid)

    # Set up
    val_old, val_new = np.zeros(num_states), np.zeros(num_states)
    store_policy = np.tile(np.nan, [max_iter, num_states])
    value_store_iter = np.tile(np.nan, [max_iter, num_states])

    # val_old = v_init

    for i in range(max_iter):
        # Policy improvement
        improved_value, improved_policy = v_greedy(val_old, u_grid, beta)

        max_diff = np.max(np.absolute(val_old - improved_value))
        # quit iterations, when convergence is achieved
        if max_diff < crit:
            break

        Q_policy = np.zeros((num_states, num_states))
        Q_policy[range(len(improved_policy)),improved_policy] = 1.0

        U_policy = u_grid[improved_policy,range(len(improved_policy))]

        # Policy evaluation with k iterations
        for _ in range(k):
            val_new = U_policy + beta * Q_policy.dot(improved_value)
                
            max_diff = np.max(np.absolute(val_new - improved_value))
            # quit iterations, when convergence is achieved
            if max_diff < crit:
                break

            improved_value = val_new
            
        val_old = improved_value
            
        #update policy
        store_policy[i,:] = improved_policy
        value_store_iter[i,:] = improved_value

    num_iter = i

    return value_store_iter, store_policy, num_iter







