from math import sqrt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

Q = 30
n = 0.013
So = 0.001
b = 10
z = 2

g=9.81

alpha=1

def ch_geo(y):
    A =b*y + z*y**2
    P= b + 2*y*(sqrt(1+z**2))
    R=A/P
    return A,P,R


def velocity(Q,A):
    return Q/A

def friction_slope(Q,n,A,R):
    return (n * Q / (A * R ** (2 / 3))) ** 2

def specific_energy(y,V, alpha=1):
    return y + alpha * V ** 2 / (2 * g)

y=5

#for the calculation of  uniform flow depth use fsolve from scipy library Sf-S0=0

def Sf_minus_S0(y):
    A, P, R = ch_geo(y[0])
    Sf = friction_slope(Q, n, A, R)
    return [Sf - So]

y_guess = 1.6
y_end = fsolve(Sf_minus_S0, [y_guess])[0]
print(f"Uniform flow depth y_n ≈ {y_end:.4f} m")

dy=-0.1
x=0
x_vals = []
y_vals = []

print(f"\n{'y (m)':<8}{'A':<10}{'R':<10}{'V':<10}{'Sf':<12}{'E':<10}{'ΔE':<10}{'Δx':<10}{'x (m)':<10}")
print("-" * 80)

while y + dy >= y_end:
    A1, P1, R1 = ch_geo(y)
    V1 = velocity(Q, A1)
    E1 = specific_energy(y, V1)
    Sf1 = friction_slope(Q, n, A1, R1)

    y2 = y + dy
    A2, P2, R2 = ch_geo(y2)
    V2 = velocity(Q, A2)
    E2 = specific_energy(y2, V2)
    Sf2 = friction_slope(Q, n, A2, R2)

    Sf_avg = (Sf1 + Sf2) / 2
    dE = E2 - E1
    dx = dE / (So - Sf_avg)
    x += dx

    x_vals.append(x)
    y_vals.append(y)

    print(f"{y:<8.2f}{A1:<10.2f}{R1:<10.2f}{V1:<10.2f}{Sf1:<12.6f}{E1:<10.5f}{dE:<10.5f}{dx:<10.2f}{x:<10.2f}")

    y = y2


plt.figure()
plt.plot(x_vals, y_vals, linestyle='-')
plt.xlabel("Distance  x (m)")
plt.ylabel("Water depth y (m)")
plt.title("Water Surface Profile ")
plt.gca().set_aspect(100)

plt.grid(True)
plt.show()







