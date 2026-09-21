import random

def scalar_product_row_col(matrix: list[list[int]]) -> int:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError(f"matrix must be square and non-empty")

    min_val = matrix[0][0]
    min_row = 0
    max_val = matrix[0][0]
    max_col = 0

    for i in range(n):
        for j in range(n):
            if matrix[i][j] < min_val:
                min_val = matrix[i][j]
                min_row = i
            if matrix[i][j] > max_val:
                max_val = matrix[i][j]
                max_col = j

    return sum(matrix[min_row][k] * matrix[k][max_col] for k in range(n))


def find_min_k(matrix: list[list[int]]) -> int:
    if not matrix or not matrix[0]:
        raise ValueError(f"matrix must be non-empty")

    result = None
    for row in matrix:
        row_max = max(row)
        k = row_max + 1
        if result is None or k < result:
            result = k

    return result


def input_matrix_keyboard() -> list[list[int]]:
    n = int(input(f"enter matrix size n: "))
    if n <= 0:
        raise ValueError(f"size must be positive")
    matrix = []
    for i in range(n):
        row = list(map(int, input(f"row {i}: ").split()))
        if len(row) != n:
            raise ValueError(f"expected {n} elements, got {len(row)}")
        matrix.append(row)
    return matrix


def generate_matrix_random(low: int, high: int, n: int) -> list[list[int]]:
    if n <= 0:
        raise ValueError(f"size must be positive")
    if low > high:
        raise ValueError(f"low must be <= high")
    return [[random.randint(low, high) for i in range(n)] for j in range(n)]


def print_matrix(matrix: list[list[int]]) -> None:
   for row in matrix:
        print(row)

m = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
print(f"scalar product:", scalar_product_row_col(m))
print(f"min K:", find_min_k(m))