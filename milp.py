# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:42:38 2023

@author: fabio
"""

import pulp
import pandas as pd
import time

# Load Market Data
market_data = pd.read_excel("Second Round Technical Question - Attachment 2 - Sao Paolo.xlsx", sheet_name="Half-hourly data")
market_3_data = pd.read_excel("Second Round Technical Question - Attachment 2 - Sao Paolo.xlsx", sheet_name="Daily data")

# Battery Specifications
max_charge_rate = 2.  # Maximum charge rate in MW
max_discharge_rate = 2.  # Maximum discharge rate in MW
max_storage = 4  # Maximum storage volume in MWh
charge_efficiency = 0.95  # Charging efficiency (0-1 scale)
discharge_efficiency = 0.95  # Discharging efficiency (0-1 scale)
max_cycles = 5000  # Maximum battery lifetime in battery cycles equivalent

# Input Variables
T = 1096 * 48
market_1_prices = market_data['Market 1 Price [£/MWh]']
market_2_prices = market_data['Market 2 Price [£/MWh]']
market_3_prices = market_3_data['Market 3 Price [£/MWh]']

st = time.time()

# Instantiate the model
model = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)

# Charge/discharge variables
charge_from_1 = {t: pulp.LpVariable(f"charge_to_1_{t}", 0, max_charge_rate) for t in range(T)}
discharge_to_1 = {t: pulp.LpVariable(f"discharge_to_1_{t}", 0, max_discharge_rate) for t in range(T)}

charge_from_2 = {t: pulp.LpVariable(f"charge_from_2_{t}", 0, max_charge_rate) for t in range(T)}
discharge_to_2 = {t: pulp.LpVariable(f"discharge_to_2_{t}", 0, max_discharge_rate) for t in range(T)}

charge_from_3 = {t: pulp.LpVariable(f"charge_from_3_{t}", 0, max_charge_rate) for t in range(T)}
discharge_to_3 = {t: pulp.LpVariable(f"discharge_to_3_{t}", 0, max_discharge_rate) for t in range(T)}

# State-of-charge (SoC) decision variables
energy_level = {t: pulp.LpVariable(f"energy_level_{t}", 0, max_storage) for t in range(T)}
discharge_allowed = {t: pulp.LpVariable(f"discharge_allowance_{t}", 0, 1, pulp.LpBinary) for t in range(T)}

# Objective function
model += pulp.lpSum([
    - (charge_from_1[t] * market_1_prices[t] * charge_efficiency) +
    (discharge_to_1[t] * market_1_prices[t] * (1/discharge_efficiency)) -
    (charge_from_2[t] * market_2_prices[t] * charge_efficiency) +
    (discharge_to_2[t] * market_2_prices[t] * (1/discharge_efficiency)) -
    (charge_from_3[t] * market_3_prices[int(t/48)] * charge_efficiency) +
    (discharge_to_3[t] * market_3_prices[int(t/48)] * (1/discharge_efficiency))
    for t in range(T)
]), "Total Revenue"

# Market 3 has daily time granularity
# Assumption: the price at day D is the same for every half-hour of day D
daily_steps = int(T/48)
# Constraints for Market 3 constant action
# Breaks daily Market 3 data into half-hour
for t_prime in range(daily_steps):
    for t in range(t_prime*48 + 1, (t_prime+1)*48):
        model += charge_from_3[t] == charge_from_3[t_prime * 48]
        model += discharge_to_3[t] == discharge_to_3[t_prime * 48]

# It is not allowed to sell/buy the same unit of power into multiple markets
for t in range(T):
    model += charge_from_1[t] + charge_from_2[t] + charge_from_3[t] <= max_charge_rate, f"Total_Charge_Rate_Constraint_at_{t}"
    model += discharge_to_1[t] + discharge_to_2[t] + discharge_to_3[t] <= max_discharge_rate, f"Total_Discharge_Rate_Constraint_at_{t}"

# SoC dynamics
model += energy_level[0] == charge_from_1[0] * charge_efficiency + charge_from_2[0] * charge_efficiency + charge_from_3[0] * charge_efficiency

for t in range(1, T):
    model += energy_level[t] == (energy_level[t - 1] + charge_from_1[t] * charge_efficiency + charge_from_2[t] * charge_efficiency + charge_from_3[t] * charge_efficiency - discharge_to_1[t] * (1 / discharge_efficiency) - discharge_to_2[t] * (1 / discharge_efficiency) - discharge_to_3[t] * (1 / discharge_efficiency))

# Preventing the battery to discharge when energy_level is 0
# Battery capacity constraints
for t in range(T):
    model += energy_level[t] <= max_storage
    model += energy_level[t] >= 0

    # If energy_level[t] > 0: discharged_allowed[t] = 1
    # If energy_level[t] = 0: discharged_allowed[t] = 0
    model += discharge_allowed[t] <= energy_level[t]
    model += discharge_to_1[t] <= max_discharge_rate * discharge_allowed[t]
    model += discharge_to_2[t] <= max_discharge_rate * discharge_allowed[t]
    model += discharge_to_3[t] <= max_discharge_rate * discharge_allowed[t]

# Preventing simultaneous charge/discharge across the three markets

# Define binary charge indicator for each timestep
charge_indicator = {t: pulp.LpVariable(f"charge_indicator_{t}", 0, 1, pulp.LpBinary) for t in range(T)}

for t in range(T):
    # If charge_indicator[t] = 1, charging is allowed and discharging is set to 0
    model += charge_from_1[t] <= max_charge_rate * charge_indicator[t]
    model += charge_from_2[t] <= max_charge_rate * charge_indicator[t]
    model += charge_from_3[t] <= max_charge_rate * charge_indicator[t]
    
    model += discharge_to_1[t] <= max_discharge_rate * (1 - charge_indicator[t])
    model += discharge_to_2[t] <= max_discharge_rate * (1 - charge_indicator[t])
    model += discharge_to_3[t] <= max_discharge_rate * (1 - charge_indicator[t])

print(">> Charging/discharging variables and constraints added successfully!")

# According to definition:
# one cycle is defined as charging up to max storage volume and then discharging all stored energy. This does not have
# to be done in one go - e.g. charging up to 75%, discharging to 0%, then charging up to 25% and discharging to 0% still
# counts as one cycle.

# Cycle count variables and constraints
cycle_count = {t: pulp.LpVariable(f"cycle_count_{t}", 0, None) for t in range(T)}
charged = {t: pulp.LpVariable(f"charged_{t}", 0, 1, pulp.LpBinary) for t in range(T)}
discharged = {t: pulp.LpVariable(f"discharged_{t}", 0, 1, pulp.LpBinary) for t in range(T)}

# Update cycle count based on charging and discharging actions
for t in range(1, T):
    total_charge_t = (charge_from_1[t] + charge_from_2[t] + charge_from_3[t])
    total_discharge_t = (discharge_to_1[t] + discharge_to_2[t] + discharge_to_3[t])

    # Updates the cycle_count for time t by adding the sum of total charge and discharge (scaled by the maximum storage
    # capacity) to the previous time step's cycle count.
    # Only adds 1 to cycle_count[t] when (total_charge_t + total_discharge_t) / max_storage equals to 1, i.e.,
    # full discharge
    model += cycle_count[t] == cycle_count[t - 1] + (total_charge_t + total_discharge_t) / max_storage
    # Ensure that the total charge and discharge at time t do not exceed their maximum rates
    model += total_charge_t <= charged[t] * max_charge_rate
    model += total_discharge_t <= discharged[t] * max_discharge_rate

# Ensure the total number of cycles does not exceed the maximum allowed
model += cycle_count[T - 1] <= max_cycles, "Max_Cycles_Constraint"

print(">> Cycle counts variables and constraints added successfully!")

# Solve the problem with optimality gap equals to 0.3
model.solve(pulp.PULP_CBC_CMD(gapRel=0.3))

# Half-hourly battery charging/discharging and SoC
charge_values_1 = {t: charge_from_1[t].varValue for t in range(T)}
discharge_values_1 = {t: discharge_to_1[t].varValue for t in range(T)}
charge_values_2 = {t: charge_from_2[t].varValue for t in range(T)}
discharge_values_2 = {t: discharge_to_2[t].varValue for t in range(T)}
charge_values_3 = {t: charge_from_3[t].varValue for t in range(T)}
discharge_values_3 = {t: discharge_to_3[t].varValue for t in range(T)}
energy_levels = {t: energy_level[t].varValue for t in range(T)}

# Calculate revenue for each market
revenue_1 = sum([-charge_values_1[t] * market_1_prices[t] * charge_efficiency +
                 discharge_values_1[t] * market_1_prices[t] / discharge_efficiency for t in range(T)])

revenue_2 = sum([-charge_values_2[t] * market_2_prices[t] * charge_efficiency +
                 discharge_values_2[t] * market_2_prices[t] / discharge_efficiency for t in range(T)])

revenue_3 = sum([-charge_values_3[t] * market_3_prices[int(t/48)] * charge_efficiency +
                 discharge_values_3[t] * market_3_prices[int(t/48)] / discharge_efficiency for t in range(T)])

# Total revenue
total_revenue = revenue_1 + revenue_2 + revenue_3
print("Total Revenue: £", total_revenue)

# Load the half-hourly charging/discharging and SoC to a dataframe and save it to a CSV file
df = pd.DataFrame(
    [
        energy_levels, charge_values_1, charge_values_2, charge_values_3,
        discharge_values_1, discharge_values_2, discharge_values_3
    ]
)
df = df.T
df.columns = ['SoC', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
df.to_csv("MILP_BEST.csv")

e = time.time()
runtime = e - st
print(f"Runtime: {runtime:.2f} seconds")

print()
