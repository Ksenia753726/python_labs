from typing import List
#var 1
def calculate_product_even_indices(arr: List[int]) -> int:
    product = 1
    for i in range(1, len(arr), 2):
        product *= arr[i]
    return product

def calculate_sum_between_zeros(arr: List[int]) -> int:
    first_zero = -1
    last_zero = -1

    for i, val in enumerate(arr):
        if val == 0:
            if first_zero == -1:
                first_zero = i
            last_zero = i

    if first_zero == -1 or last_zero == -1 or first_zero + 1 >= last_zero:
        return 0

    total = 0
    for i in range(first_zero + 1, last_zero):
        total += arr[i]
    return total

def transform_array(arr: List[int]) -> List[int]:
    negatives = []
    non_negatives = []
    for x in arr:
        if x < 0:
            negatives.append(x)
        else:
            non_negatives.append(x)
    return negatives + non_negatives

def main() -> None:
    arr = [3, 0, -2, 4, 0, 5, -1, 6]

    product_even = calculate_product_even_indices(arr)
    sum_between = calculate_sum_between_zeros(arr)
    transformed = transform_array(arr)

    print(product_even)
    print(sum_between)
    print(transformed)

if __name__ == "__main__":
    main()
