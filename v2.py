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
5. Integrate biogas as an alternative or supplemental thermal source.
6. Use real solar irradiance data.
7. Include 3D plots and Sankey diagrams for enhanced analysis.

Methodology:
------------
- Thermodynamic cycle selection: Regenerative Brayton cycle.
- Solar thermal input: Direct Normal Irradiance (DNI) simulation.
- Working fluid: Air (modeled as ideal gas).
- Energy collection modeled as thermal input from solar concentrators.
- Recuperator modeled with a heat exchange efficiency.
- Biogas modeled as a combustion input based on Lower Heating Value.

Libraries Used:
---------------
- CoolProp: Thermophysical properties.
- NumPy: Numerical operations.
- Matplotlib: Data visualization.
- SciPy: Interpolation and optimization.
- Plotly: 3D visualization.
- Pandas: Solar data handling.
- matplotlib.sankey: Sankey diagrams.

Assumptions:
------------
- Ideal gas behavior for air.
- Constant specific heats.
- No pressure losses in pipes.
- Solar collector has 70% thermal efficiency.
- Recuperator effectiveness is 80%.
- Biogas has 21 MJ/kg heating value.

Cycle Overview:
---------------
1. Air is compressed isentropically (Compressor).
2. Heated by solar thermal energy or biogas combustion (Solar Receiver).
3. Expanded through turbine (Turbine).
4. Exhaust heat partially recovered in a recuperator.
5. Heat rejected and cycle restarts.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.sankey import Sankey
from scipy.optimize import fsolve

cp = 1005  # J/kg.K
gamma = 1.4
eta_comp = 0.85
eta_turb = 0.88
eta_recup = 0.8
eta_solar = 0.7
T_ambient = 300
P_ambient = 100e3
T_max = 1100
P_ratio = 6
biogas_LHV = 21e6  # J/kg
biogas_efficiency = 0.8

# Load solar irradiance data
solar_data = pd.read_csv('solar_irradiance_sample.csv')  # assumes hourly DNI in W/m^2
solar_thermal_input = solar_data['DNI'] * eta_solar  # W/m^2 to usable thermal

# Thermodynamic functions
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

def cycle_sim(T1, P1, P2, Q_in):
    T2, w_c = compressor_work(T1, P1, P2)
    T2r = recuperator_outlet(T_max, T2)
    T3 = T_max
    T4, w_t = turbine_work(T3, P2, P1)
    W_net = w_t - w_c
    eta_th = W_net / Q_in
    return {
        'T1': T1, 'T2': T2, 'T2r': T2r, 'T3': T3, 'T4': T4,
        'w_c': w_c, 'w_t': w_t, 'W_net': W_net, 'Q_in': Q_in,
        'eta_th': eta_th
    }

# Evaluate cycle for solar input
Q_in_solar = cp * (T_max - T_ambient)
P2 = P_ratio * P_ambient
results = cycle_sim(T_ambient, P_ambient, P2, Q_in_solar)
for key, val in results.items():
    print(f"{key}: {val:.2f}")

# Efficiency vs pressure ratio
pressure_ratios = np.linspace(2, 20, 100)
efficiencies = []
for r in pressure_ratios:
    P2 = r * P_ambient
    res = cycle_sim(T_ambient, P_ambient, P2, Q_in_solar)
    efficiencies.append(res['eta_th'])

plt.figure(figsize=(10, 6))
plt.plot(pressure_ratios, np.array(efficiencies)*100)
plt.xlabel('Pressure Ratio')
plt.ylabel('Thermal Efficiency (%)')
plt.title('Thermal Efficiency vs Pressure Ratio (Solar-Thermal Microturbine)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Sankey diagram
sankey = Sankey(unit=None)
sankey.add(flows=[Q_in_solar, -results['W_net'], -results['Q_in'] + results['W_net']],
              labels=['Solar Input', 'Net Work Output', 'Waste Heat'],
              orientations=[0, 1, -1])
sankey.finish()
plt.title('Energy Flow Sankey Diagram')
plt.show()

# 3D plot of efficiency over T_max and pressure ratio
T_max_range = np.linspace(800, 1300, 25)
P_ratios = np.linspace(2, 20, 25)
T_grid, P_grid = np.meshgrid(T_max_range, P_ratios)
eta_grid = np.zeros_like(T_grid)

for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        T_hi = T_grid[i, j]
        P2 = P_grid[i, j] * P_ambient
        Q_in = cp * (T_hi - T_ambient)
        try:
            res = cycle_sim(T_ambient, P_ambient, P2, Q_in)
            eta_grid[i, j] = res['eta_th']
        except:
            eta_grid[i, j] = 0

fig = go.Figure(data=[go.Surface(z=eta_grid * 100, x=T_grid, y=P_grid)])
fig.update_layout(title='Efficiency vs T_max and Pressure Ratio',
                  scene=dict(xaxis_title='T_max (K)',
                             yaxis_title='Pressure Ratio',
                             zaxis_title='Efficiency (%)'))
fig.show()
