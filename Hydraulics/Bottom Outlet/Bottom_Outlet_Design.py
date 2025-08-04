import matplotlib.pyplot as plt
import pandas as pd
from math import *



'''
filename=str(input('Pls specify the path to your excel file containing Bottom outlet design values: '))
'''
filename = 'bottom_outlet.xlsx'

df = pd.read_excel(filename, sheet_name='Data')
variables = {}
for index, row in df.iterrows():
    variables[row['Definition']] = row['Data_for_the_variables']
h =     variables['hmax'] - variables['z2']

def Arfunc(a,b,h):
    Ar=a*(h**b)
    return float('{:.4g}'.format(Ar))


Vmax = float('{:.4g}'.format(sqrt(2 * 9.81 * h)))
Armax = Arfunc(variables['a'],variables['b'],variables['hmax'])

def Qs(u,Ar):
    return float('{:.4g}'.format((1/3600)*u*Ar))

Qsmax = Qs(variables['Us'],Arfunc(variables['a'],variables['b'],variables['hmax']))



Qbmax = min(variables['Q0'] + Qsmax, variables['Ql'])

A2 = float('{:.4g}'.format(Qbmax / Vmax))

hmax = int(variables['hmax'])
hmin = int(variables['hmin'])

Ar_list = []
qs_list = []
q0plusqs_list = []
qballowed_list = []
hres_list = []

for i in range(hmax, hmin - 1, -1):
    hres = i
    hres_list.append(hres)
    Ar = Arfunc(variables['a'], variables['b'], i)
    Ar_list.append(Ar)
    qs = Qs(variables['Us'], Arfunc(variables['a'], variables['b'], i))
    qs_list.append(qs)
    q0plusqs = variables['Q0'] + qs
    q0plusqs_list.append(q0plusqs)
    qballowed = min(Qbmax, q0plusqs)
    qballowed_list.append(qballowed)

y_b = variables['y/b']


def f(Dh):
    log_inner = log((variables['e'] / (3.7 * Dh)))
    value = 1.325 / ((log_inner) ** 2)
    return value
def descap(g, Kent, A1, f1, L1, Dh1, Ktr, A2, f2, L2, Dh2, Kg, hres, z2, y2, ):
    numerator = 2 * 9.81 * (hres - z2 - y2)
    denominator = (
            Kent / A1 ** 2 +
            f1 * L1 / (Dh1 * A1 ** 2) +
            Ktr / A2 ** 2 +
            f2 * L2 / (Dh2 * A2 ** 2) +
            Kg / A2 ** 2 +
            1 / A2 ** 2
    )
    return sqrt(numerator / denominator)
# design capacity calculations

initial_b2 = float('{:.4g}'.format(sqrt(A2 * y_b)))

b2 = ceil(initial_b2 * 10) / 10

descap_dict={} #dictionary to store the lists
b2_dict={}

for j in range(0,3):
    descap_list = []
    for i in range(hmax, hmin - 1, -1):
        y2 = b2 * variables['y/b']
        b1 = float('{:.4g}'.format((b2 / variables['b2/b1'])))
        R1 = b1 / 2
        y1 = b1 * variables['y/b']
        A1 = float('{:.4g}'.format((b1 * y1 + (pi * (R1 ** 2)) / 2)))
        P1 = float('{:.4g}'.format((b1 + 2 * y1 + pi * R1)))
        Rh1 = float('{:.4g}'.format((A1 / P1)))
        A2 = float('{:.4g}'.format((b2 * y2)))
        Dh1 = float('{:.4g}'.format((4 * Rh1)))
        P2 = float('{:.4g}'.format((2 * (b2 + y2))))
        Rh2 = float('{:.4g}'.format((A2 / P2)))
        Dh2 = float('{:.4g}'.format((4 * Rh2)))
        f1 = f(Dh1)
        f2 = f(Dh2)
        descapacity = float('{:.4g}'.format(descap(9.81, variables['Kent'], A1, f1, variables['L1'], Dh1, variables['Ktr'], A2, f2,
                             variables['L2'], Dh2, variables['Kg'], i, variables['z2'], y2)))
        descap_list.append(descapacity)
    plt.plot(descap_list, hres_list, label=f'$b_2$={b2} m')
    descap_dict[j] = descap_list
    b2_dict[j]=b2
    b2 += 0.1


plt.plot(q0plusqs_list,hres_list)
plt.xlabel("Q (m³/s)", fontsize=12)
plt.ylabel("h (m)", fontsize=12)
plt.title("Optimization for the Design Discharge", fontsize=14)
plt.legend()
plt.grid()
plt.show()



# now lets check if the Qdes < Qb

results = {}

for key, values in descap_dict.items():
    all_less_than= True
    for i,value in enumerate(values):
        if value >= qballowed_list[i]:
            all_less_than= False
            break
        else:
            continue
    results[key]=all_less_than

for key, result in results.items():
    if result:
        print(f"All values in '{key}' are less than the corresponding allowable values.")
    else:
        print(f"Some values in '{key}' are NOT less than the corresponding allowable values.")


passing_keys = [key for key, result in results.items() if result]

if len(passing_keys) == len(descap_dict):
    print(f"All lists pass. Using '{passing_keys[-1]}' for design.")
    design_list = descap_dict[passing_keys[-1]]
    b2_design=b2_dict[passing_keys[-1]]
elif len(passing_keys) > 0:
    selected_key = passing_keys[-1]
    print(f"Using '{selected_key}' for design.")
    design_list = descap_dict[selected_key]
    b2_design=b2_dict[selected_key]
else:
    raise ValueError("None of the lists pass the conditions. Cannot proceed with design.")







df_new = pd.DataFrame({
    "hres": hres_list,
    "Ar":Ar_list,
    'Qs':qs_list,
    'Qs+Q0':q0plusqs_list,
    '(Qb)allowed':qballowed_list,
    'b2_1':descap_dict[0],
    'b2_2':descap_dict[1],
    'b2_3':descap_dict[2],
    'final_design':design_list
})
with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
    df_new.to_excel(writer, sheet_name="Obtained values", index=False)


def delta_volume(a, b, h1, h2):
    delta_vol = (a / (b + 1)) * (h2 ** (b + 1) - h1 ** (b + 1))
    return delta_vol


delta_V_list = []
h_mid_list = []
Q_b_list = []
Q_diff_list = []
delta_t_list = []
cumulative_t_list = []

cumulative_time = 0

for i in range(0,len(hres_list) -2,2 ):
    h1 = hres_list[i]
    h2 = hres_list[i + 2]
    hmid = (h1 + h2) / 2
    Q_b = design_list[i+1]
    Q_diff = variables['Q0'] - Q_b
    delta_V = (variables['a'] / (variables['b'] + 1)) * (h2 ** (variables['b'] + 1) - h1 ** (variables['b'] + 1))
    delta_t = (delta_V / Q_diff)/3600
    if delta_t < 0:
        break
    cumulative_time += delta_t
    delta_V_list.append(round(delta_V / 1e7, 3))  # Convert to 10^7 m³
    h_mid_list.append(hmid)
    Q_b_list.append(Q_b)
    Q_diff_list.append(Q_diff)
    delta_t_list.append(round(delta_t, 2))
    cumulative_t_list.append(round(cumulative_time, 2))


df_drawdown=pd.DataFrame({
    "ΔV (x10^7 m³)": delta_V_list,
    "h_mid (m)": h_mid_list,
    "Q_b (m³/s)": Q_b_list,
    "Q₀ - Q_b (m³/s)": Q_diff_list,
    "Δt (hrs)": delta_t_list,
    "T = ΣΔt (hrs)": cumulative_t_list
})

with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
    df_drawdown.to_excel(writer, sheet_name="Drawdown", index=False)


plt.plot(cumulative_t_list,h_mid_list)
plt.ylabel("h (m)", fontsize=12)
plt.xlabel("T (hr)", fontsize=12)
plt.title("Drawdown time", fontsize=14)
plt.grid()
plt.xlim(left=0)
plt.ylim(bottom=variables['hmin'], top=max(h_mid_list))
plt.show()

#now lets do the gate computations

a0=b2_design*variables['y/b'] #buraya dikkat et düzelt

delta_h_e_max=(hmax-variables['z2']-a0) -((design_list[0])**2)/(2*9.81*(a0*b2_design)**2)

K_total=delta_h_e_max/design_list[0]**2
print(K_total)


area_ratios = []


for ratio in range(10, 0, -1):
    area_ratios.append(ratio / 10)

areas = [a0 * ratio for ratio in area_ratios]


area_ratios_list = []
areas_list = []
C_c_list = []
C_c_a_list = []
Q_assumed_list = []
delta_h_e_list = []
Q_computed_list = []


def compute_Q(a, Q_assumed):

    C_c = 0.8 + 0.2 * (a / a0) ** 4


    C_c_a = C_c * a

    # Compute Δh_e
    delta_h_e = K_total * Q_assumed ** 2

    # Compute Q_computed
    Q_computed = C_c * a * b2_design *sqrt(2 * 9.81 * (h - delta_h_e - C_c_a))

    return C_c, C_c_a, delta_h_e, Q_computed



tolerance=0.00000001




for a in areas:
    Q_assumed = design_list[0]

    while True:

        C_c, C_c_a, delta_h_e, Q_computed = compute_Q(a, Q_assumed)


        if abs(Q_assumed - Q_computed) < tolerance:
            break


        Q_assumed = Q_computed


    area_ratios_list.append(a / a0)
    areas_list.append(a)
    C_c_list.append(round(C_c, 3))
    C_c_a_list.append(round(C_c_a, 3))
    Q_assumed_list.append(round(Q_assumed, 2))
    delta_h_e_list.append(round(delta_h_e, 2))
    Q_computed_list.append(round(Q_computed, 2))


df_gate = pd.DataFrame({
    "a/a0": area_ratios_list,
    "a": areas_list,
    "C_c": C_c_list,
    "C_c * a": C_c_a_list,
    "Q_assumed": Q_assumed_list,
    "Δh_e": delta_h_e_list,
    "Q_computed": Q_computed_list
})


plt.subplot(121)

plt.plot(area_ratios_list,C_c_list)
plt.ylabel("$C_c$", fontsize=12)
plt.xlabel("$a/a_0$ ", fontsize=12)
plt.title("Gate lip contraction coefficient", fontsize=14)
plt.grid()
plt.xlim(left=0, right=1)
plt.ylim(bottom=C_c_list[-1]-0.1, top=1)

plt.subplot(122)

plt.plot(Q_computed_list,area_ratios_list)
plt.ylabel("$a/a_0$ ", fontsize=12)
plt.xlabel('Q (m³/s)', fontsize=12)
plt.title('Gate-openning discharge curve', fontsize=14)
plt.grid()

plt.show()

with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
    df_gate.to_excel(writer, sheet_name="Gate", index=False)







