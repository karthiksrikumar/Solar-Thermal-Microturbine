# Title: Python-Based Thermodynamic Simulation of a Solar-Thermal Microturbine for Decentralized Energy Production

"""
Problem Statement:
-------------------
Remote and low-income regions lack reliable access to centralized electricity infrastructure. Fossil-fuel-based backup generators used in such areas are inefficient and polluting. A sustainable, decentralized, and renewable energy system is needed. This project proposes a hybrid solar-thermal microturbine system, modeled entirely in Python, capable of simulating thermodynamic performance under variable environmental conditions.

Objectives:
-----------
1. Simulate a solar-thermal Brayton cycle using Python.
2. Incorporate thermal storage and recuperation for efficiency.
3. Visualize thermal and power outputs under varying solar irradiance.
4. Allow parametric tuning for optimization studies.

Methodology:
------------
- Thermodynamic cycle selection: Regenerative Brayton cycle.
- Solar thermal input: Direct Normal Irradiance (DNI) simulation.
- Working fluid: Air (modeled as ideal gas).
- Energy collection modeled as thermal input from solar concentrators.
- Recuperator modeled with a heat exchange efficiency.

Libraries Used:
---------------
- CoolProp: Thermophysical properties.
- NumPy: Numerical operations.
- Matplotlib: Data visualization.
- SciPy: Interpolation and optimization.

Assumptions:
------------
- Ideal gas behavior for air.
- Constant specific heats.
- No pressure losses in pipes.
- Solar collector has 70% thermal efficiency.
- Recuperator effectiveness is 80%.

Cycle Overview:
---------------
1. Air is compressed isentropically (Compressor).
2. Heated by solar thermal energy (Solar Receiver).
3. Expanded through turbine (Turbine).
4. Exhaust heat partially recovered in a recuperator.
5. Heat rejected and cycle restarts.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants and Parameters
cp = 1005  # J/kg.K, specific heat at constant pressure (air)
gamma = 1.4  # Ratio of specific heats
eta_comp = 0.85  # Compressor efficiency
eta_turb = 0.88  # Turbine efficiency
eta_recup = 0.8  # Recuperator effectiveness
eta_solar = 0.7  # Solar thermal efficiency
T_ambient = 300  # K
P_ambient = 100e3  # Pa
T_max = 1100  # K, max solar heater temp
P_ratio = 6  # Pressure ratio of compressor

# Functions for isentropic relations
def isentropic_temp_out(T_in, P_in, P_out, gamma):
    return T_in * (P_out / P_in)**((gamma - 1) / gamma)

def compressor_work(T1, P1, P2):
    T2s = isentropic_temp_out(T1, P1, P2, gamma)
    T2 = T1 + (T2s - T1) / eta_comp
    w_c = cp * (T2 - T1)
    return T2, w_c

def turbine_work(T3, P3, P4):
    T4s = isentropic_temp_out(T3, P3, P4, gamma)
    T4 = T3 - eta_turb * (T3 - T4s)
    w_t = cp * (T3 - T4)
    return T4, w_t

def recuperator_outlet(T4, T2):
    return T2 + eta_recup * (T4 - T2)

def net_work(w_t, w_c):
    return w_t - w_c

def cycle_sim(T1, P1, P2):
    T2, w_c = compressor_work(T1, P1, P2)
    T2r = recuperator_outlet(T_max, T2)  # pre-heated by exhaust
    Q_in = cp * (T_max - T2r)
    T3 = T_max
    T4, w_t = turbine_work(T3, P2, P1)
    W_net = net_work(w_t, w_c)
    eta_th = W_net / Q_in
    return {
        'T1': T1, 'T2': T2, 'T2r': T2r, 'T3': T3, 'T4': T4,
        'w_c': w_c, 'w_t': w_t, 'W_net': W_net, 'Q_in': Q_in,
        'eta_th': eta_th
    }

# Run simulation
P2 = P_ratio * P_ambient
results = cycle_sim(T_ambient, P_ambient, P2)

# Display results
for key, val in results.items():
    print(f"{key}: {val:.2f}")

# Parametric analysis: Efficiency vs Pressure Ratio
pressure_ratios = np.linspace(2, 20, 100)
efficiencies = []

for r in pressure_ratios:
    P2 = r * P_ambient
    try:
        res = cycle_sim(T_ambient, P_ambient, P2)
        efficiencies.append(res['eta_th'])
    except:
        efficiencies.append(0)

plt.figure(figsize=(10, 6))
plt.plot(pressure_ratios, np.array(efficiencies)*100)
plt.xlabel('Pressure Ratio')
plt.ylabel('Thermal Efficiency (%)')
plt.title('Thermal Efficiency vs Pressure Ratio (Solar-Thermal Microturbine)')
plt.grid(True)
plt.tight_layout()
plt.show()
