from useful_functions import knapsack

from dataclasses import dataclass

from typing import List, Dict, Tuple

import numpy as np

import numba

import tqdm

@dataclass
class Library:
    id_: int

    number_of_books: int
    number_of_days_for_signup: int
    # the number of books that can be shipped from library j
    # to the scanning facility per day, once the library is signed up.
    library_throughput_per_day: int

    book_ids: List[int]

@dataclass
class ProblemInput:
    number_of_days_total: int
    book_scores: List[int]
    libraries: List[Library]

@dataclass
class LibraryOutput:
    id_: int
    book_ids: List[int]

@dataclass
class ProblemSolution:
    libraries: List[LibraryOutput]


def read_input(name: str) -> ProblemInput:
    with open('data/in/' + name + '.txt', 'r') as f:
        B, L, D = f.readline().strip('\n').split(' ')
        B, L, D = list(map(int, [B, L, D]))

        book_scores = f.readline().strip('\n').split(' ')
        book_scores = list(map(int, book_scores))

        libraries = []

        for l in range(L):
            N, T, M = f.readline().strip('\n').split(' ')
            N, T, M = list(map(int, [N, T, M]))

            book_ids = f.readline().strip('\n').split(' ')
            book_ids = list(map(int, book_ids))

            library = Library(l, N, T, M, book_ids)

            libraries.append(library)

        return ProblemInput(D, book_scores, libraries)


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


def write_output(name: str, input_: ProblemInput, sol: ProblemSolution):
    with open('data/out/' + name + '.out', 'w') as f:
        f.write(str(len(sol.libraries)))
        f.write('\n')
        for lib in sol.libraries:
            f.write(f'{lib.id_} {len(lib.book_ids)}\n')
            f.write(' '.join([str(s) for s in lib.book_ids]))
            f.write('\n')


def main():
    problem_names = ['a_example']

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
# solve_problem_fast(np.array([100]), np.array([0] * 100), np.zeros((100, 100)))

# # Just to test input output works
print(read_input('a_example'))
solution = ProblemSolution([
    LibraryOutput(1, [5, 2, 3]),
    LibraryOutput(0, [0, 1, 2, 3, 4])
])
write_output('a_example', None, solution)

# main()