import matplotlib.pyplot as plt
import numpy as np

nofdrops=[37,28,20]
mofcontwetsoil=[16.28,19.39,20.11,10.22,10.21]
mofcontdrysoil=[11.44,13.27,13.59,8.87,8.9]
mofcontainer=[3.9,4.17,4.07,3.93,4.7]
#the fallowing above are given values

mofmoisture=[]
mofdrysoil=[]
moisturecontent_lst=[]

for i in range(0,5):
    massofmoisture=mofcontwetsoil[i]-mofcontdrysoil[i]
    mofmoisture.append(massofmoisture)
    massofdrysoil=mofcontdrysoil[i]-mofcontainer[i]
    mofdrysoil.append(massofdrysoil)
    moisturecontent=(massofmoisture/massofdrysoil)*100
    moisturecontent_lst.append(moisturecontent)
print(f'Mass of moisture: {mofmoisture}')
print(f'Mass of dry soil: {mofdrysoil}')
print(f'Moisture content: {moisturecontent_lst}')

x=np.array(nofdrops)
y=np.array(moisturecontent_lst)
a,b=np.polyfit(x,y,1)
plt.scatter(x,y)
plt.ylim(0,100)
plt.plot(x, a*x+b, color='steelblue', linestyle='--', linewidth=2)
plt.text(20, 40, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
plt.ylabel('$Water Content (percent)$',fontsize=22)
plt.xlabel('$No. of Drops$', fontsize=22)



