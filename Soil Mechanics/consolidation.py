import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

D=6.35
A=(math.pi*(D**2))/4 #cm^2
H=1.88 #cm

G_s = 2.69
p_w=1

elapsed_time_lst = [0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 240, 456, 863, 1649, 2968]
total_deformation_lst = [2.103, 2.121, 2.131, 2.152, 2.184, 2.237, 2.295, 2.395, 2.537, 2.689, 2.853, 2.966, 3.002,
                         3.028, 3.042]
stress = 640

m_ring = 68.81
m_ring_w = 167.65
m_ring_d = 131.56
w_f = 0.3

m_water=m_ring_w - m_ring_d
m_soil=m_ring_d - m_ring

w_i = (m_water) / (m_soil)

V=A*H
V_s= m_soil/(G_s*p_w)
V_v=V-V_s
e=V_v/V_s
S_r=(m_water/p_w)/V_v
two_H_0=V_s/A
print('Values of the first table:', end=' ')
print('{:.2f}'.format(V),end=', ')
print('{:.2f}'.format(V_s),end=', ')
print('{:.2f}'.format(V_v),end=', ')
print('{:.2f}'.format(e),end=', ')
print('{:.3f}'.format(S_r),end=', ')
print('{:.2f}'.format(two_H_0))

x1=np.array(elapsed_time_lst)
y1=np.array(total_deformation_lst)
plt.figure(1)
plt.plot(x1,y1)
plt.scatter(x1,y1, color='red')
plt.xscale('log')
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')

plt.title('Deformation vs Log Time', fontsize= 24, color='red',y=1.09)
plt.ylabel('d(mm)',fontsize=16)
plt.xlabel('Time(min)', fontsize=16)
plt.grid(True, which="both", ls="-")
yticks = np.arange(2, 3.1, 0.05)
plt.yticks(yticks)


sqrt_elapsed_time_lst=[]
for i in elapsed_time_lst:
    sqrt_elapsed_time_lst.append(math.sqrt(i))

x2=np.array(sqrt_elapsed_time_lst)
y2=np.array(total_deformation_lst)
plt.figure(2)

cs = CubicSpline(x2, y2)
# Generate a smooth curve
x_smooth = np.linspace(x2.min(), x2.max(), 100)
y_smooth = cs(x_smooth)
plt.plot(x_smooth,y_smooth)



plt.scatter(x2,y2, color='red')
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')

plt.title('Deformation vs Sqrt Time', fontsize= 24, color='red',y=1.09)
plt.ylabel('d(mm)',fontsize=16)
plt.xlabel('Sqrt Time(min)', fontsize=16)
plt.grid()
yticks = np.arange(2, 3.1, 0.05)
plt.yticks(yticks)
xticks=np.arange(0,60,5)
plt.xticks(xticks)

plt.show()

# by hand we calculated delta h at 640 consolşdation pressure

d=8.135
T_v_fifty=0.196
t_50_sec=58*60
C_v='{:.5f}'.format((T_v_fifty*(d**2))/t_50_sec)
print(f'Cv casagrande = {C_v} mm^2/sec')


T_v_ninty=0.848
t_90_sec=(17.5**2)*60
C_v='{:.5f}'.format((T_v_ninty*(d**2))/t_90_sec)
print(f'Cv taylors = {C_v} mm^2/sec')

# going back to automation

d_100_=2.97
consolidation_pressure_lst=[0,20,40,80,160,320,640,1280]
l=len(consolidation_pressure_lst)
delta_H_lst=[0,0,0.129,0.328,0.693,1.587,d_100_,5.261]
H_lst=[]
equivalent_void_H_lst=[]
e_lst=[]

for i in range(l):
    H1=float('{:.5f}'.format(H*10-delta_H_lst[i]))
    equivalent_void_H=float('{:.5f}'.format(H1-two_H_0*10))
    e=float('{:.5f}'.format(equivalent_void_H/(two_H_0*10)))
    H_lst.append(H1)
    equivalent_void_H_lst.append(equivalent_void_H)
    e_lst.append(e)
print(f'Delta H: {H_lst}')
print(f'Equivalent Void Height: {equivalent_void_H_lst}')
print(f'e: {e_lst}')


x3=np.array(consolidation_pressure_lst[1:l])
y3=np.array(e_lst[1:l])
plt.figure(3)

cs = CubicSpline(x3, y3)

# Generate a smooth curve
x_smooth = np.linspace(x3.min(), x3.max(), 100)
y_smooth = cs(x_smooth)




plt.plot(x_smooth,y_smooth)
plt.scatter(x3,y3, color='red')
plt.xscale('log')
plt.grid()
plt.title('e vs log sigma v_effective', fontsize= 24, color='red',y=1.09)
plt.ylabel('e',fontsize=16)
plt.xlabel('log sigma v_effective', fontsize=16)


x4=np.array(consolidation_pressure_lst)
y4=np.array(e_lst)
plt.figure(4)

cs = CubicSpline(x4, y4)
# Generate a smooth curve
x_smooth = np.linspace(x4.min(), x4.max(), 100)
y_smooth = cs(x_smooth)

plt.plot(x_smooth,y_smooth)
plt.scatter(x4,y4, color='red')
plt.title('e vs sigma v_effective', fontsize= 24, color='red',y=1.09)
plt.ylabel('e',fontsize=16)
plt.xlabel('sigma v_effective', fontsize=16)
plt.grid()

plt.show()

t_50_sec_lst=[2710,2920,3130,3280,t_50_sec,3650]
consolidation_pressure_lst=consolidation_pressure_lst[2:]
l=len(consolidation_pressure_lst)
avg_pres_lst=['-']
avg_H_lst=['-']
Cv_lst=['-']

for i in range(l-1):
    avg_pres=float('{:.5f}'.format((consolidation_pressure_lst[i]+consolidation_pressure_lst[i+1])/2))
    avg_H= float('{:.5f}'.format((H_lst[i+2] + H_lst[i+3])/2))
    Cv=float('{:.5f}'.format((T_v_fifty*((avg_H)/2)**2)/t_50_sec_lst[i+1]))

    Cv_lst.append(Cv)
    avg_H_lst.append(avg_H)
    avg_pres_lst.append(avg_pres)


print(f'Avg Pressure: {avg_pres_lst}')
print(f'Average Height:{avg_H_lst}')
print(f'Cv: {Cv_lst}')


x5=np.array(avg_pres_lst[1:l])
y5=np.array(Cv_lst[1:l])
plt.figure(5)

cs = CubicSpline(x5, y5)
# Generate a smooth curve
x_smooth = np.linspace(x5.min(), x5.max(), 100)
y_smooth = cs(x_smooth)

plt.plot(x_smooth,y_smooth)
plt.scatter(x5,y5, color='red')
plt.xscale('log')
plt.title('Cv vs Log Average Pressure', fontsize= 24, color='red',y=1.09)
plt.ylabel('Cv (mm^2/sec) ',fontsize=16)
plt.xlabel('Log Average Pressure', fontsize=16)
plt.grid()

plt.show()

