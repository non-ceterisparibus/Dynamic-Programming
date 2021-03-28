import matplotlib as mpl
import matplotlib.pyplot as plt
import random

def plot_dynamics(d1_grid, value, iter, lines, colormap):
    """
    Parameters
    ----------
    d1_grid : vector, of length n (ndim=1)
    

    """
    random.seed(3000)
    iter_vector = random.sample(range(0,iter-1), lines)
    iter_vector.sort()
    iter_vector.append(iter-1)  # add last iteration
    
    plt.subplots(figsize = (8, 6))
    x = d1_grid
    
    #Ploting
    cmap = plt.cm.get_cmap(colormap)
    norm = mpl.colors.SymLogNorm(2, vmin=iter_vector[0], vmax=iter_vector[-1])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for j in range(lines+1):
        y = value[iter_vector[j],:]
        plt.plot(x, y, linewidth=2, alpha=0.6, color=cmap(norm(iter_vector[j])), label=iter_vector[j])

def plot_value_function(k_grid, val_store_iter, num_iter,lines):
    "Plot the value function "

    plot_dynamics(k_grid, val_store_iter, num_iter, lines, colormap = 'Reds')
   
    plt.ylabel("Value function")
    plt.xlabel("Capital today")
    
    plt.show()

def plot_transition_dynamics(T, k_sim, num_iter, lines):
    "Plot the transition dynamics of capital "

    plot_dynamics(T, k_sim, num_iter, lines, colormap = "Purples")
   
    plt.ylabel("Capital stock, k_t")
    plt.xlabel("time, t")

    plt.text(0.5, 0.5, '$k_{t+1} = g(k_t)$ from various $k_0$', 
    horizontalalignment='center', verticalalignment='center'
    )
    plt.show()

def plot_policy_function(k_grid, val_store_iter, num_iter):
    "Plot the value function "

    #Iteration vector
    iter_vector = range(0,num_iter)

    plt.subplots(figsize = (8, 6))
    x = k_grid
    
    #Ploting
    cmap = plt.cm.get_cmap("Reds")
    norm = mpl.colors.SymLogNorm(2, vmin=iter_vector[0], vmax=iter_vector[-1])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for j in range(num_iter):
        y = val_store_iter[j,:]
        plt.plot(x, y, linewidth=2, alpha=0.6, color=cmap(norm(iter_vector[j])), label=iter_vector[j])
   
    plt.ylabel("Value function")
    plt.xlabel("Capital today")

    cbar = plt.colorbar(sm, ticks= iter_vector, format=mpl.ticker.ScalarFormatter(), 
                    shrink=1.0, fraction=0.1, pad=0)
    
    plt.show()
    
def plot_capital_current_utility(u_grid, k_grid):
    """

    """
    num_states = len(k_grid)

    random.seed(3000)
    randidx = random.sample(range(num_states), 2)

    plt.subplots(figsize=(6,6))
    for id in randidx:
        y = u_grid[:,id]
        plt.plot(k_grid, y, linewidth=2, alpha=0.8)

    plt.ylabel("$U(c_t)$")
    plt.xlabel("Choice of next-period capital, $k_(t+1)$")
    plt.legend(
            labels=['capital state, k_t = '+ str(round(k_grid[id],4)) for id in randidx],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
        )

def plot_bellman_equation(u_grid, k_grid, beta, value_store_iter_v, num_iter_v):
    """

    """
    num_states = len(k_grid)

    random.seed(3000)
    randidx = random.sample(range(num_states), 2)

    plt.subplots(figsize=(6,6))

    for id in randidx:
        y = u_grid[:,id] + beta*value_store_iter_v[num_iter_v-1,:]
        plt.plot(k_grid, y, linewidth=2, alpha=0.8)

    plt.ylabel("$U(c_t) + beta* V_{t+1}(k_{t+1})$")
    plt.xlabel("Choice of next-period capital, $k_(t+1)$")
    plt.legend(
            labels=['capital state, k_t = '+ str(round(k_grid[id],4)) for id in randidx],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
        )
