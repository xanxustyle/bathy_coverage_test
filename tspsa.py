from functools import reduce
from scipy.spatial.distance import pdist
from random import randint, random, sample
import math
import numpy as np
from numba import jit, prange, njit


@njit(fastmath=True, parallel=True)
def compute_path_travel_distance(waypts):
    distance = 0
    for i in prange(0, len(waypts) - 1):  # "parallel programming is hard"
        distance += np.linalg.norm(waypts[i] - waypts[i + 1])  # euclidean distance
    return distance


@njit(fastmath=True)
def reverse_random_sublist(lst,waypts):
    # I read online that this was much better than a random permutation for getting convergence
    new_list = lst.copy()
    waypts_len = len(waypts) - 1
    start = randint(0, waypts_len)
    end = randint(start, waypts_len)
    new_list[start:end + 1] = new_list[start:end + 1][::-1]
    return new_list


@njit
def random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return list(sample(pool, r))


@njit(fastmath=True)
def acceptance_probability(old_cost, new_cost, temperature):
    res = math.exp((old_cost - new_cost) / temperature)
    return res


@njit(fastmath=True)
def tsp_sa(waypts):
    old_cost = [compute_path_travel_distance(waypts)]
    temperature = 1.0
    min_temperature = 1e-10
    alpha = 0.95
    # best_solution = None
    solution = waypts
    while temperature > min_temperature:
        for iteration in range(1, 500):
            # canidate = random_permutation(solution, r = len(waypts)) #Not NEARLY as good as the other one
            canidate = reverse_random_sublist(solution,waypts)
            new_cost = compute_path_travel_distance(canidate)
            ap = acceptance_probability(old_cost[-1], new_cost, temperature)
            if ap > random():
                solution = canidate
                old_cost.append(new_cost)
                # if len(old_cost) > 10 & iteration > 100:
                #     if (old_cost[-10] - old_cost[-1]) < 50:
                #         print('break')
                #         break
        temperature = temperature * alpha
    return solution, compute_path_travel_distance(solution)
