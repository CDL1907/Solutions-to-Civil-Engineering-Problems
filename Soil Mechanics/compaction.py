import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

total_massof_sample=5000
Gs=2.65
per_coarser_than_nine_point_five=0.16
w_coarse=0
Gscoarse=2.65
height_of_mold=11.6
volume_of_mold=929.4
ro_w=1

mofcontwetsoil=[64,69,72,75.5,81]
mofcontdrysoil=[59,62.5,64,66,69]
mofcontainer=[32,31,30,31,30]

M2=[5720,5880,6000,5890,5760]
M1=[4180,4180,4180,4180,4180]


#the fallowing above are given values

mofmoisture=[]
mofdrysoil=[]
moisturecontent_lst=[]
w_corrected_lst=[]


mass_of_compacted_soil_lst=[]
bulk_density_lst=[]
dry_density_lst=[]
corrected_dry_density_lst=[]

for i in range(0,5):
    massofmoisture=mofcontwetsoil[i]-mofcontdrysoil[i]
    mofmoisture.append(massofmoisture)

    massofdrysoil=mofcontdrysoil[i]-mofcontainer[i]
    mofdrysoil.append(massofdrysoil)

    moisturecontent=(massofmoisture/massofdrysoil)*100
    moisturecontent_lst.append(moisturecontent)

    w_corrected= per_coarser_than_nine_point_five*w_coarse+ (1-per_coarser_than_nine_point_five)*moisturecontent
    w_corrected_lst.append(w_corrected)


    mass_of_compacted_soil= M2[i]-M1[i]
    mass_of_compacted_soil_lst.append(mass_of_compacted_soil)

    bulk_density=mass_of_compacted_soil/volume_of_mold
    bulk_density_lst.append(bulk_density)

    dry_density=(100*bulk_density)/(100+moisturecontent)
    dry_density_lst.append(dry_density)

    corrected_dry_density=(dry_density*Gscoarse*ro_w)/(dry_density*per_coarser_than_nine_point_five + (Gscoarse*ro_w*(1-per_coarser_than_nine_point_five)))
    corrected_dry_density_lst.append(corrected_dry_density)



print(f'Mass of moisture: {mofmoisture}')
print(f'Mass of dry soil: {mofdrysoil}')
print(f'Moisture content: {moisturecontent_lst}')
print(f'Corrected moisture content: {w_corrected_lst}')
print(f'Mass of compacted soil: {mass_of_compacted_soil_lst}')
print(f'Bulk density {bulk_density_lst}')
print(f'Dry density {dry_density_lst}')
print(f'corrected dry density {corrected_dry_density_lst}')


df = pd.DataFrame({'x': w_corrected_lst,
                   'y': corrected_dry_density_lst})

model2 = np.poly1d(np.polyfit(df.x, df.y, 2))
polyline = np.linspace(15, 27, 50)
plt.scatter(df.x,df.y)
plt.plot(polyline, model2(polyline), color='red')
plt.ylabel('$Dry Density(g/cm3)$',fontsize=22)
plt.xlabel('$Moisture Content (percent)$', fontsize=22)
x = np.linspace(15, 27, 100)
y=2.65/(1+(0.0331*x))
plt.plot(x,y, color='orange', linestyle='--', linewidth=2)
x = np.linspace(15, 27, 100)
y=2.65/(1+(x*0.0294))
plt.plot(x,y, color='green', linestyle='--', linewidth=2)
x = np.linspace(15, 27, 100)
y=2.65/(1+(x*0.0265))
plt.plot(x,y, color='grey', linestyle='--', linewidth=2)
plt.legend(['Corrected Dry Density vs Moisture Content','Corrected Dry Density vs Moisture Content Best Curve','Sr=0.8','Sr=0.9','Sr=1'])

plt.xlim(0,27)
plt.ylim(0,2.5)

print(f'{model2}')

plt.show()
