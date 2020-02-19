from useful_functions import knapsack

from dataclasses import dataclass

from typing import List, Dict, Tuple

import numpy as np

import numba

import tqdm


@dataclass
class ProblemInput:
    pass


@dataclass
class ProblemSolution:
    pass


# TODO: Code me
def read_input(name: str) -> ProblemInput:
    # abc = f.readline().strip('\n').split(' ')
    # abc = list(map(int, abc))
    # these two will be used often here

    with open(name + '.in', 'r') as f:
        return ProblemInput()


# Do not write types as numba will infer them
# These will probably be some lists or nd-arrays
# Can return multiple outputs
@numba.jit(nopython=True)
def solve_problem_fast(param1, param2, param3):
    return np.array([1, 2, 3])


# Will invoke out1, out2 = solve_problem_fast(input_.param1, input_.param2 ...)
# return ProblemSolution(out1, out2)
# If you use tqdm use it here, not inside solve_problem_fast
def solve_problem(input_: ProblemInput) -> ProblemSolution:
    return ProblemSolution()


# TODO: Code me
def write_output(name: str, input_: ProblemInput, sol: ProblemSolution):
    # will often use f.write() and f.writelines()
    with open(name + '.out', 'w') as f:
        pass


def main():
    problem_names = ['a_example', 'b_easy']

    for name in problem_names:
        input_ = read_input(name)

        # For debugging
        if name == 'a_example':
            print('Input ', input_)

        sol = solve_problem(input_)

        if name == 'a_example':
            print('Solution ', sol)

        write_output(name, input_, sol)


# Test to see numba works
solve_problem_fast(np.array([100]), np.array([0] * 100), np.zeros((100, 100)))
