import numpy as np
import matplotlib.pyplot as plt

def compute_force_vector(L, num_elements):
    h = L / num_elements
    F = np.full(num_elements, h)
    F[-1] = h / 2 + 1
    return F

def assemble_global_stiffness_matrix(L, num_elements):
    h = L / num_elements
    coefficient = 100 / h**2
    global_K = np.zeros((num_elements + 1, num_elements + 1))
    for i in range(num_elements):
        lower_limit = i * h
        upper_limit = (i + 1) * h
        integral_result = h + (1 / 3) * ((upper_limit**2) - (lower_limit**2))
        element_stiffness = coefficient * integral_result
        K_e = element_stiffness * np.array([[1, -1], [-1, 1]])
        global_K[i:i + 2, i:i + 2] += K_e
    return global_K

def apply_boundary_conditions(global_K, force_vector):
    reduced_K = global_K[1:, 1:]
    reduced_F = force_vector
    return reduced_K, reduced_F

def solve_fem(L, num_elements):
    global_K = assemble_global_stiffness_matrix(L, num_elements)
    force_vector = compute_force_vector(L, num_elements)
    reduced_K, reduced_F = apply_boundary_conditions(global_K, force_vector)
    print("Reduced Stiffness Matrix (K):")
    print(reduced_K)
    print("\nReduced Force Vector (F):")
    print(reduced_F)
    displacements = np.linalg.solve(reduced_K, reduced_F)
    return displacements

L = 3
num_elements = int(input("Enter the number of elements: "))

displacements = solve_fem(L, num_elements)
print("\nDisplacements (U):")
print(displacements)

def exact_solution(x):
    return 0.0825 * np.log(100 + (200 / 3) * x) - 0.015 * x - 0.3798

x_exact = np.linspace(0, L, 100)
# Compute exact displacements
u_exact = exact_solution(x_exact)

displacements_full = np.insert(displacements, 0, 0)
x_fem = np.linspace(0, L, num_elements + 1)

# Plot both FEM and exact solution
plt.figure(figsize=(10, 6))

plt.plot(x_fem, displacements_full, 'o-', label="FEM Solution")
plt.plot(x_exact, u_exact, '-', label="Exact Solution")
plt.xlabel("x (length)")
plt.ylabel("Displacement (U)")
plt.title("Displacement: FEM vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()
# This code implements a finite element method (FEM) to solve a problem defined by a stiffness matrix and a force vector.
