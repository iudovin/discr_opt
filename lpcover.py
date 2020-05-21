import time
import numpy as np
from scipy.optimize import linprog


def read_input():
    n_columns, n_rows = map(int, input().split())
    row_weights = []
    cover_list = []
    for _ in range(n_rows):
        row_covered_columns = [int(x) for x in input().split()]
        row_weights.append(row_covered_columns[0])
        cover_list.append(row_covered_columns[1:])
    return n_columns, n_rows, row_weights, cover_list


def prepro(n_columns, n_rows, cover_list):
    matrix = np.zeros((n_rows, n_columns), dtype=int)
    for r in range(n_rows):
        for c in cover_list[r]:
            matrix[r,c] = 1
    return matrix


def proba_round(matrix, proba):
    n_rows, n_cols = matrix.shape
    x = [0] * n_rows
    
    def is_covered():
        return all([np.dot(matrix.T[c], x) >= 1 for c in range(n_cols)])

    while True:
        for i in range(n_rows):
            if x[i] == 0:
                is_up = np.random.rand() < proba[i]
                if is_up:
                    x[i] = 1
        if is_covered():
            break
    return x


eval_start = time.time()

n_columns, n_rows, row_weights, cover_list = read_input()
matrix = prepro(n_columns, n_rows, cover_list)

exact_res = linprog(row_weights, A_ub=-matrix.T, b_ub=-np.ones(n_columns))

eval_end = time.time()

m = n_rows
ans = [1] * n_rows
timeout = 25 + eval_start - eval_end
timeout_start = time.time()
while time.time() < timeout_start + timeout:
    rnd = proba_round(matrix, exact_res.x)
    if len(rnd) < m:
        m = len(rnd)
        ans = rnd

for i in range(n_rows):
    if ans[i] == 1:
        print(i+1, end=' ')
print()