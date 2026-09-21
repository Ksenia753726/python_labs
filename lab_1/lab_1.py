import math
def find_ln(x : float, e : float) -> float :
    sum = 0
    n = 1
    temp = e
    while abs(temp) >= e:
        temp = -(x ** n) / n
        sum += temp
        n += 1
    return sum


x = float(input("enter x [-1, 1): "))
e = float(input("enter e (0, 1): "))
try:
    if x < -1 or x >= 1 :
        raise ValueError("wrong x")
    if e <= 0 or e >= 1 :
        raise  ValueError("wrong e")
    my_ln = find_ln(x, e)
    exact = math.log(1 - x)
    print(f"my value of ln(1 - x) : {my_ln}")
    print(f"original value of ln(1 - x) : {exact}")
except ValueError as e :
    print(e)