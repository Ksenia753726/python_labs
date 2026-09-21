import math


def is_prime(n : int) -> bool :
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def count_digits(n : int) -> int :
    if n == 0:
        return 1
    c = 0
    while n > 0:
        c += 1
        n //= 10
    return c


def are_digits_unique(x : int) -> bool :
    flag = True
    por = count_digits(x)
    if por > 10 :
       flag = False
    else :
        list_c = [0] * 10
        for i in range(por) :
            temp = x % 10
            list_c[temp] += 1
            x //= 10
        print(list_c)
        for numb in list_c :
            if numb > 1 :
                flag = False
                break
    return flag

def find_min_dig(x : int) -> int :
    min = x % 10
    while x != 0 :
        temp = x % 10
        if temp < min :
            min = temp
        x = int(x / 10) 
    return min

def remove_digits_divisible_by_min(x : int) -> int :
    temp = 0
    min = find_min_dig(x)
    result = 0
    while x != 0  :
        if(x % min != 0) :
           temp = temp * 10 + x % 10
        x = int(x / 10) 
    while temp != 0 :
        result = result * 10 + temp % 10
        temp = int(temp / 10)
    return result

def most_frequent_digit(n : int) -> int :
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    counts = [0] * 10
    while n > 0:
        counts[n % 10] += 1
        n //= 10
    best = 0
    for d in range(1, 10):
        if counts[d] > counts[best]:
            best = d
    return best


def add_most_frequent_sides(n : int) -> int :
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    d = most_frequent_digit(n)
    length = count_digits(n)
    return d * (10 ** (length + 1)) + n * 10 + d


def add_central_digit_sides(n : int) -> int :
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    length = count_digits(n)
    if length % 2 == 0:
        return n
    mid = length // 2
    tmp = n
    for i in range(mid):
        tmp //= 10
    central = tmp % 10
    return central * (10 ** (length + 1)) + n * 10 + central


def fibonacci_sum_below(n : int) -> int :
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 1, 1
    total = 0
    while total + a <= n:
        total += a
        a, b = b, a + b
    return total


def prime_factorize(n: int) -> list[int]:
    if n < 2:
        raise ValueError("n must be >= 2")
    factors = []
    while n > 1:
        div = 2
        while n % div != 0:
            div += 1
        factors.append(div)
        n //= div
    return factors


try :
    print(are_digits_unique(1224))
    print(remove_digits_divisible_by_min(235679))
    print(add_most_frequent_sides(122334))
    print(add_central_digit_sides(12345))
    print(fibonacci_sum_below(20)) 
    print(prime_factorize(360))
except ValueError as e :
    print(e)