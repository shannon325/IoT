import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt

# Load testing results
testing_results = np.loadtxt('TestingResults.txt')

# Verify if the last column is an anomaly label
last_column = testing_results[:, -1]
abnormal_indices = np.where(last_column == 1)[0]

# Load abnormal price curves
abnormal_price_curves = testing_results[abnormal_indices, :-1]  # Exclude the label column

# Load user and task information from Excel file
tasks_df = pd.read_excel('IMSE7143CW1Input.xlsx', sheet_name=0)

def schedule_energy_for_user(price_curve, user_tasks_df):
    # Initialize the problem
    problem = LpProblem("Energy_Scheduling", LpMinimize)
    
    # Create decision variables for each task and time slot
    time_slots = range(24)
    task_vars = {(task, t): LpVariable(f"{task}_at_{t}", 0, None, cat='Continuous')
                 for task in user_tasks_df['User & Task ID']
                 for t in time_slots}
    
    # Objective function: minimize total cost
    problem += lpSum(task_vars[task, t] * price_curve[t]
                     for task in user_tasks_df['User & Task ID']
                     for t in time_slots), "Total_Cost"
    
    # Add constraints
    for _, row in user_tasks_df.iterrows():
        task_id = row['User & Task ID']
        ready_time = int(row['Ready Time'])
        deadline = int(row['Deadline'])
        max_energy_per_hour = row['Maximum scheduled energy per hour']
        energy_demand = row['Energy Demand']
        
        # Ensure the task is only scheduled within its time window
        problem += lpSum(task_vars[task_id, t] for t in time_slots if ready_time <= t <= deadline) == energy_demand, f"Demand_{task_id}"
        
        # Adhere to maximum scheduled energy per hour
        for t in range(ready_time, deadline + 1):
            problem += task_vars[task_id, t] <= max_energy_per_hour, f"MaxEnergy_{task_id}_{t}"
    
    # Solve the problem
    solver = PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    
    # Collect results
    energy_usage = np.zeros(24)
    for t in time_slots:
        energy_usage[t] = sum(task_vars[task, t].varValue for task in user_tasks_df['User & Task ID'])
    
    return energy_usage

# Calculate total energy scheduling for each abnormal price curve
for i, price_curve in enumerate(abnormal_price_curves):
    total_energy_usage = np.zeros(24)
    for user in tasks_df['User & Task ID'].apply(lambda x: x.split('_')[0]).unique():
        user_tasks_df = tasks_df[tasks_df['User & Task ID'].str.startswith(user)]
        energy_usage = schedule_energy_for_user(price_curve, user_tasks_df)
        total_energy_usage += energy_usage
    
        # Plot total energy usage
        plt.figure(figsize=(10, 6))
        plt.bar(range(24), total_energy_usage, width=1.0, align='edge', color='skyblue')
        plt.title(f'TestingResults No.{abnormal_indices[i]+1}\nTotal Energy Usage for All Users - Abnormal Case {i+1}')
        plt.xlabel('Hour')
        plt.ylabel('Total Energy Usage')
        plt.xticks(range(25), labels=[f'{h}' for h in range(24)] + ['24'])
        plt.grid(True)
        plt.savefig(f'./energy_scheduling/total_energy_usage_case_{i+1}.png')
        plt.close()

        # Plot price curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(24), price_curve, marker='o', color='orange')
        plt.title(f'TestingResults No.{abnormal_indices[i]+1}\nPrice Curve - Abnormal Case {i+1}')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.xticks(range(24))
        plt.grid(True)
        plt.savefig(f'./energy_scheduling/price_curve_case_{i+1}.png')
        plt.close()
