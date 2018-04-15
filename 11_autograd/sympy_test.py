from sympy import *
x,t,z, nu = symbols('x t z nu')
result = diff(sin(x)*exp(x),x)
print(result)