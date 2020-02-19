from useful_functions import knapsack

from dataclasses import dataclass

from typing import List, Dict, Tuple

import numpy as np

import numba

import tqdm


@dataclass
class ProblemInput:
    max_slices: int
    n_pizza_types: int
    slices_per_pizza: List[int]


@dataclass
class ProblemSolution:
    chosen_pizzas: List[int]


def read_input(name: str) -> ProblemInput:
    # abc = f.readline().strip('\n').split(' ')
    # abc = list(map(int, abc))
    # these two will be used often here

    name = f'practice_round_files/{name}'

    with open(name + '.in', 'w') as f:
        max_slices, n_pizza_types = f.readline().strip('\n').split(' ')

        slices_per_pizza = f.readline().strip('\n').split(' ')
        slices_per_pizza = [int(x) for x in slices_per_pizza]

        return ProblemInput(max_slices, n_pizza_types, slices_per_pizza)


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
    slices_per_pizza = list(enumerate(input_.slices_per_pizza))
    slices_per_pizza = sorted(slices_per_pizza, key=lambda x: x[1], reverse=True)

    chosen_items = []

    w = input_.max_slices

    j = 0
    while w >= 0:
        while j < len(slices_per_pizza) and slices_per_pizza[j][1] >= w:
            j += 1

        try:
            w -= slices_per_pizza[j][1]
            if w < 0:
                break

            chosen_items.append(slices_per_pizza[j][0])
            j += 1

        except IndexError:
            break

    chosen_items = sorted(chosen_items)

    return ProblemSolution(chosen_items)


# Note: input_ was not used here but in more complex problems it could be useful
def write_output(name: str, input_: ProblemInput, sol: ProblemSolution):
    # will often use f.write() and f.writelines()
    name = f'practice_round_files/{name}'

    with open(name + '.out', 'w') as f:
        f.write(str(len(sol.chosen_pizzas)))
        f.write('\n')
        f.write(' '.join([str(i) for i in sol.chosen_pizzas]))
        f.write('\n')


def main():
    problem_names = ['a_example', 'b_small', 'c_medium', 'd_quite_big',' e_also_big']

    for name in problem_names:
        input_ = read_input(name)

        # For debugging
        if name == 'a_example':
            print('Input ', input_)

        sol = solve_problem(input_)

        if name == 'a_example':
            print('Solution ', sol)

        write_output(name, input_, sol)

main()

# Test to see numba works
# solve_problem_fast(np.array([100]), np.array([0] * 100), np.zeros((100, 100)))
