import numpy as np
import matplotlib.pyplot as plt


q = 0.0312
q2=0.0476
b = 0.075
g = 9.81

# Flow depth range (vertical axis)
y = np.linspace(0.01, 0.16, 500)

# Specific Energy and Force (horizontal axis)
E = y + q**2 / (2 * g * y**2)
F = (q**2 * b) / (g * y) + (b * y**2) / 2



measured_y = np.array([0.151, 0.01925, 0.026, 0.07529, 0.08,0.0291])
measured_E = measured_y + q**2 / (2 * g * measured_y**2)
measured_F = (q**2 * b) / (g * measured_y) + (b * measured_y**2) / 2


b=0.049

E2 = y + q2**2 / (2 * g * y**2)
F2 = (q2**2 * b) / (g * y) + (b * y**2) / 2

y_q2_measured = 0.0608
E_q2_point = y_q2_measured + q2**2 / (2 * g * y_q2_measured**2)
F_q2_point = (q2**2 * b) / (g * y_q2_measured) + (b * y_q2_measured**2) / 2




colors = ['red', 'orange', 'purple', 'brown', 'cyan','grey']





plt.figure()
plt.plot(E, y, color='blue', label=f'Specific Energy (q = {q})', linewidth=2)
plt.plot(E2, y, color='green', label=f'Specific Energy (q = {q2}) contraction', linewidth=2)

for i in range(len(measured_y)):
    plt.scatter(measured_E[i], measured_y[i], color=colors[i], label=f'y = {measured_y[i]:.5f} m', zorder=5)

plt.scatter(E_q2_point, y_q2_measured, color='magenta', marker='x',
            label=f'y = {y_q2_measured:.5f} m (q={q2}) contraction', zorder=5)


plt.title('Specific Energy vs. Flow Depth')
plt.xlabel('Specific Energy E (m)')
plt.ylabel('Flow Depth y (m)')
plt.grid(True)
plt.legend()


plt.figure()
plt.plot(F, y, color='blue', linestyle='--', label=f'Specific Force (q = {q})', linewidth=2)
plt.plot(F2, y, color='green', linestyle='--', label=f'Specific Force (q = {q2}) contraction ', linewidth=2)
for i in range(len(measured_y)):
    plt.scatter(measured_F[i], measured_y[i], color=colors[i], label=f'y = {measured_y[i]:.5f} m', zorder=5)


plt.scatter(F_q2_point, y_q2_measured, color='magenta', marker='x',
            label=f'y = {y_q2_measured:.5f} m (q={q2}) contraction', zorder=5)

plt.title(f'Specific Force vs. Flow Depth')
plt.xlabel('Specific Force F (m³)')
plt.ylabel('Flow Depth y (m)')
plt.grid(True)
plt.legend()


plt.show()
