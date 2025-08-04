from math import sqrt

g=9.81
alpha=1
Q=30
n=0.013
b=10
z=2
So=0.001

# the geometry calculations will be for trapezoidal channel


def area(y):
    return b*y + z*y**2

def perimeter(y):
    return b + 2 * y * sqrt(1+z**2)

def hydraulic_radius(y):
    return area(y)/perimeter(y)

def dA_dy(y):
    return b + 2 * z * y

def dR_dy(y):
    A = area(y)
    P=perimeter(y)
    dA=dA_dy(y)
    dP=2*sqrt(1+z**2)
    return (dA*P-A*dP)/(P**2)   #chain rule for division

def friction_slope(Q, n, A, R):
    return (n * Q / (A * R ** (2/3))) ** 2



def F(y2):
    A2 = area(y2)
    R2 = hydraulic_radius(y2)
    Sf2 = friction_slope(Q, n, A2, R2)
    E2_term = y2 + alpha * Q ** 2 / (2 * g * A2 ** 2)
    return E2_term + 0.5 * Sf2 * dx + z2-z1 - E1 + 0.5 * Sf1 * dx

def dF_dy(y2):
    A2 = area(y2)
    R2 = hydraulic_radius(y2)
    B2 = dA_dy(y2)
    dR2 = dR_dy(y2)
    Sf2 = friction_slope(Q, n, A2, R2)
    term1 = 1 - alpha * Q ** 2 * B2 / (g * A2 ** 3)
    term2 = Sf2 * (B2 / A2 + (2 / 3) * dR2 / R2)
    return term1 - dx * term2


results = []
y1 = 5.0
z1 = 0.0
dx = -1000
x1=0
for i in range(4):
    A1 = area(y1)
    R1 = hydraulic_radius(y1)
    V1 = Q / A1
    Sf1 = friction_slope(Q, n, A1, R1)
    E1 = y1 + alpha * Q ** 2 / (2 * g * A1 ** 2)

    z2 = z1 + So *-dx
    dy_dx = So - Sf1
    y2_guess = y1 + dy_dx * dx

    print(f"Initial guess for y2: {y2_guess:.4f} m")


    tolerance = 1e-4
    max_iter = 100
    y2 = y2_guess

    x2=x1+dx

    for i in range(max_iter):
        f = F(y2)
        df = dF_dy(y2)
        y2_new = y2 - f / df

        print(f"Iter {i+1}: y2 = {y2:.6f}, f = {f:.6e}, df = {df:.6e}")

        if abs(y2_new - y2) < tolerance:
            print(f"\n✅ Converged in {i+1} iterations: y2 = {y2_new:.6f} m")
            y2 = y2_new
            break

        y2 = y2_new


    print(f"\nFinal computed y2 = {y2:.6f} m")

    # to store the values we need to recalculate the values and store them in the results array

    A2 = area(y2)
    P2 = perimeter(y2)
    R2 = A2 / P2
    R_13 = R2 ** (1 / 0.75)
    V2 = Q / A2
    V2_2g = V2 ** 2 / (2 * g)
    Sf2 = friction_slope(Q, n, A2, R2)
    S_avg = 0.5 * (Sf1 + Sf2)
    hf = S_avg * abs(dx)
    E2 = y2 + V2_2g
    H_total = E2 + z2
    diff = abs((E1 + z1) - (H_total + hf))


    results.append([
        x2, y2, A2, P2, R2, R_13, V2, V2_2g, z2, E2, Sf2, S_avg, dx, hf, H_total
    ])

    x1 = x2
    y1 = y2             #changing the values so that we are ready for the next step
    z1 = z2




headers = ["x", "y", "A", "P", "R", "R^1.33", "V", "V^2/2g", "z", "E", "Sf", "Sf_avg", "dx", "hf", "E+z"]
print("\n" + "-" * 170)
print(" | ".join(f"{h:<10}" for h in headers))
print("-" * 170)
for row in results:
    print(" | ".join(f"{val:<10.4f}" if isinstance(val, float) else f"{val:<10}" for val in row))

