import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema
"""
filename=str(input(("Enter the name of the file:")))
df=pd.read_csv(filename)
height=(int(input("Enter the height of the specimen:")))
radius=(int(input("Enter the radius of the specimen:")))
"""
df=pd.read_csv('lab_1_values.csv')
height=130
radius=50

nrow , ncol=df.shape
lvdt_avg_empty = [None]*nrow #mm
time_lst_frame=df['Time']
force_lst_frame=df['Axial_Force']
axial_cod_frame=df['Axial_COD']
time_lst=[]
force_lst=[]
axial_cod_lst_c=[]
for i in range(nrow):
    lvdt_avg_empty[i]=(df['LVDT1'][i] + df['LVDT2'][i])/2
    time_lst.append(time_lst_frame[i])
    force_lst.append(force_lst_frame[i])
    axial_cod_lst_c.append(axial_cod_frame[i]*0.44)

lvdt_avg=lvdt_avg_empty


generalastl=[] #FULL AXIAL STRAIN LIST
generalasl=[]  #FULL AXIAL STRESS LIST
for i in range(nrow):
    general_axial_stress = force_lst[i] / (math.pi * ((radius) ** 2))
    general_axial_strain = lvdt_avg[i] / height
    generalasl.append(general_axial_stress)
    generalastl.append(general_axial_strain)

def axial_force_time_graph():
    plt.plot(time_lst,force_lst)
    plt.ylabel('Force(N)')
    plt.xlabel('Time(s)')
    plt.title('Axial Force vs Time Graph')
    plt.grid()

def slopeofforcetime(ind1,ind2):
    t1,f1=time_lst[ind1],force_lst[ind1]
    t2,f2=time_lst[ind2],force_lst[ind2]
    slp=(f2-f1)/(t2-t1)
    return f'Slope of force-time graph for the {ind1}-{ind2} is',slp

def svsst(x,y):
    astl = []  # axialstrainlist
    asl = []  # axialstresslist
    for i in range(x, y):
        axial_stress = force_lst[i] / (math.pi * ((radius) ** 2))
        axial_strain = lvdt_avg[i] / height
        asl.append(axial_stress)
        astl.append(axial_strain)

    plt.plot(astl,asl)
    plt.ylabel('Axial Stress(N/mm**2)')
    plt.xlabel('Axial Strain')
    plt.title('Axial Stress vs Strain Graph')
    plt.grid()

    stress1,stress2=asl[0],asl[-1]
    strain1,strain2=astl[0],astl[-1]
    E=(stress2-stress1)/(strain2-strain1)
    print('Modulus of elasticity is:', E, 'N/(mm**2)')

def localminandmax():
    fl_array=np.array(force_lst)
    local_max_array=argrelextrema(fl_array,np.greater)
    local_min_array=argrelextrema(fl_array,np.less)
    M=len(local_max_array[0])
    N=len(local_min_array[0])
    localmaxindexlist=[]
    localminindexlist=[]
    for i in range(M):
        localmaxindexlist.append(local_max_array[0][i])
    for j in range(N):
        localminindexlist.append(local_min_array[0][j])
    print('Index for the local maximums are:', localmaxindexlist)
    print('Index for the local minimums are:', localminindexlist)
    sgmaxi = localmaxindexlist[0]
    sgmini = int((sgmaxi-localminindexlist[1])*0.4+localminindexlist[1]) #secant method is used
    print(f"Suggested index for the calculation of Young's Modulus is: {sgmaxi}-{sgmini}")

def tstvsast(x,y):
    global generalastl
    tstl=[]
    for i in range(x,y):
        transverse_strain=(axial_cod_lst_c[i])/(radius*2)
        tstl.append(transverse_strain)
    plt.plot(generalastl[x:y],tstl)
    plt.ylabel('Transverse Strain')
    plt.xlabel('Axial Strain')
    plt.title('Transverse Strain vs Axial Strain')
    plt.grid()
    transversestrain1,transversestrain2=tstl[0],tstl[-1]
    axialstrain1,axialstrain2=generalastl[x:y][0],generalastl[x:y][-1]
    v=-((transversestrain2-transversestrain1)/(axialstrain2-axialstrain1))
    print(f"Poisson's Ratio for the {x}-{y} index range is: {v}")

df['Stress']=generalasl
df['Strain']=generalastl
df.to_csv("Data_Table_Lab1.csv", index=False)


















