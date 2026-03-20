import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Pipeline Optimization", layout="wide")

st.title("Pipeline System Optimization")
st.markdown("Two-variable nonlinear optimization of pipe diameter and pump efficiency.")

# inputs
st.sidebar.header("Parameters")

rho = st.sidebar.number_input("Density (kg/m3)", value=1000.0, format="%.2f")
mu = st.sidebar.number_input("Viscosity (Pa.s)", value=0.001, format="%.4f")
L = st.sidebar.slider("Pipe Length (m)", 100, 5000, 1000)
Q = st.sidebar.slider("Flow Rate (m3/s)", 0.01, 0.2, 0.05)
Ce_input = st.sidebar.slider("Electricity Cost (₹/kWh)", 1.0, 20.0, 8.0)
hours = st.sidebar.slider("Operating Hours per Year", 1000, 10000, 8000)

a = st.sidebar.slider("Pipe Cost Coefficient", 5000, 50000, 15000)
k = st.sidebar.slider("Pump Cost Coefficient", 10000, 1000000, 500000)
gamma_eff = st.sidebar.slider("Pump Cost Exponent", 1.0, 3.0, 2.0)

include_minor = st.sidebar.checkbox("Include Minor Losses")
K_minor = st.sidebar.slider("Minor Loss Coefficient (K)", 0.0, 10.0, 2.0)

mode = st.sidebar.radio("Mode", ["Optimization Mode", "Manual Mode"])

Ce = Ce_input / 1000

def total_cost(vars):
    D, eta = vars

    if D <= 0 or eta <= 0:
        return 1e12

    v = 4*Q / (np.pi*D**2)
    Re = rho*v*D / mu

    if Re <= 0:
        return 1e12

    f = 0.316 * (Re**(-0.25))
    deltaP_major = f * (L/D) * (rho*v**2/2)
    deltaP_minor = K_minor * (rho*v**2/2) if include_minor else 0
    deltaP = deltaP_major + deltaP_minor

    pipe_cost = a * D**2 * L
    energy_cost = (deltaP*Q/eta) * hours * Ce
    pump_cost = k * (eta**gamma_eff)

    return pipe_cost + energy_cost + pump_cost

def total_cost_scaled(vars):
    return total_cost(vars) / 1e6

bounds = [(0.1, 1.0), (0.5, 0.9)]

if mode == "Optimization Mode":
    method = st.sidebar.selectbox("Method", ["L-BFGS-B", "SLSQP"])
    result = minimize(total_cost_scaled, x0=[0.3, 0.75], bounds=bounds, method=method)
    D_opt, eta_opt = result.x
    min_cost = total_cost(result.x)
else:
    D_opt = st.sidebar.slider("Diameter (m)", 0.1, 1.0, 0.3)
    eta_opt = st.sidebar.slider("Efficiency", 0.5, 0.9, 0.75)
    min_cost = total_cost([D_opt, eta_opt])
    method = "Manual"

# results
st.subheader("Results")

col1, col2, col3 = st.columns(3)
col1.metric("Diameter (m)", f"{D_opt:.4f}")
col2.metric("Efficiency", f"{eta_opt:.4f}")
col3.metric("Total Cost (₹)", f"{min_cost:,.0f}")

st.write(f"Mode: **{mode}** | Method: **{method}**")

# cost breakdown
v = 4*Q / (np.pi*D_opt**2)
Re = rho*v*D_opt / mu
f = 0.316 * (Re**(-0.25))
deltaP_major = f * (L/D_opt) * (rho*v**2/2)
deltaP_minor = K_minor*(rho*v**2/2) if include_minor else 0
deltaP = deltaP_major + deltaP_minor

pipe_cost = a * D_opt**2 * L
energy_cost = (deltaP*Q/eta_opt) * hours * Ce
pump_cost = k * (eta_opt**gamma_eff)

st.subheader("Cost Breakdown")
st.write({
    "Pipe Cost (₹)": round(pipe_cost, 0),
    "Energy Cost (₹)": round(energy_cost, 0),
    "Pump Cost (₹)": round(pump_cost, 0)
})

# plots
colA, colB = st.columns(2)

with colA:
    st.subheader("Cost vs Diameter")
    D_vals = np.linspace(0.1, 1.0, 100)
    cost_vals = [total_cost([D, eta_opt]) for D in D_vals]
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(D_vals, cost_vals)
    ax1.axvline(D_opt, linestyle='--')
    ax1.set_xlabel("Diameter")
    ax1.set_ylabel("Total Cost")
    st.pyplot(fig1)

with colB:
    st.subheader("Cost vs Efficiency")
    eta_vals = np.linspace(0.5, 0.9, 100)
    cost_eta = [total_cost([D_opt, e]) for e in eta_vals]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(eta_vals, cost_eta)
    ax2.axvline(eta_opt, linestyle='--')
    ax2.set_xlabel("Efficiency")
    ax2.set_ylabel("Total Cost")
    st.pyplot(fig2)

# 3d surface
st.subheader("3D Cost Surface")

D_grid = np.linspace(0.1, 1.0, 40)
eta_grid = np.linspace(0.5, 0.9, 40)
D_mesh, eta_mesh = np.meshgrid(D_grid, eta_grid)

Z = np.zeros_like(D_mesh)
for i in range(len(D_grid)):
    for j in range(len(eta_grid)):
        Z[j, i] = total_cost([D_mesh[j, i], eta_mesh[j, i]])

fig3 = plt.figure(figsize=(6, 4))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(D_mesh, eta_mesh, Z, cmap='viridis')
ax3.set_xlabel("Diameter")
ax3.set_ylabel("Efficiency")
ax3.set_zlabel("Cost")
st.pyplot(fig3)

# sensitivity
st.subheader("Sensitivity to Electricity Cost")

Ce_values = np.linspace(2, 15, 10)
opt_D_list = []

for c in Ce_values:
    Ce_temp = c / 1000
    def temp_cost(vars):
        D, eta = vars
        v = 4*Q / (np.pi*D**2)
        Re = rho*v*D / mu
        f = 0.316 * (Re**(-0.25))
        deltaP = f * (L/D) * (rho*v**2/2)
        return a*D**2*L + (deltaP*Q/eta)*hours*Ce_temp + k*(eta**gamma_eff)

    res = minimize(lambda x: temp_cost(x)/1e6, x0=[0.3, 0.75], bounds=bounds, method="L-BFGS-B")
    opt_D_list.append(res.x[0])

fig4, ax4 = plt.subplots(figsize=(5, 4))
ax4.plot(Ce_values, opt_D_list)
ax4.set_xlabel("Electricity Cost (₹/kWh)")
ax4.set_ylabel("Optimal Diameter")
st.pyplot(fig4)
