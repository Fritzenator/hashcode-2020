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
    number_of_days: int
    number_of_books_shippable: int

    book_ids: List[int]


@dataclass
class ProblemInput:
    days: int
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

        return ProblemInput(int(D), book_scores, libraries)


def write_output(name: str, input_: ProblemInput, sol: ProblemSolution):
    with open('data/out/' + name + '.out', 'w') as f:
        f.write(str(len(sol.libraries)))
        f.write('\n')
        for lib in sol.libraries:
            f.write(f'{lib.id_} {len(lib.book_ids)}\n')
            f.write(' '.join([str(s) for s in lib.book_ids]))
            f.write('\n')


import heapq


def solve_problem(input_: ProblemInput) -> ProblemSolution:
    solution = []

    scanned_books = set()
    libraries_score = {
        lib.id_: (input_.days / lib.number_of_days * lib.number_of_books / lib.number_of_days)
        for lib in input_.libraries
    }

    libraries_heap = []

    for lib_id, score in libraries_score.items():
        heapq.heappush(libraries_heap, (-score, lib_id))

    libraries_index = {
        library.id_: library
        for library in input_.libraries
    }

    d = 0
    while d < input_.days and libraries_heap:
        _, lib_id = heapq.heappop(libraries_heap)
        lib_out = LibraryOutput(id_=lib_id, book_ids=[])
        lib = libraries_index.get(lib_id)

        d += lib.number_of_days

        books_heap = []
        for book_id in lib.book_ids:
            book_score = input_.book_scores[book_id]
            heapq.heappush(books_heap, (-book_score, book_id))

        n = (input_.days - d) * lib.number_of_books_shippable
        books_added = 0
        while books_added < n:
            if not books_heap:
                break

            _, book_id = heapq.heappop(books_heap)

            if book_id not in scanned_books:
                scanned_books.add(book_id)
                lib_out.book_ids.append(book_id)
                books_added += 1
        if books_added > 0:
            solution.append(lib_out)

    return ProblemSolution(libraries=solution)


def main():
    problem_names = [
        'a_example',
        'b_read_on',
        'c_incunabula',
        'd_tough_choices',
        'e_so_many_books',
        'f_libraries_of_the_world'
    ]

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
#
# solution = ProblemSolution([
#     LibraryOutput(1, [5, 2, 3]),
#     LibraryOutput(0, [0, 1, 2, 3, 4])
# ])
# write_output('a_example', None, solution)
