import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_path_accel = Path(r"C:\Users\Torezhan\projects\FallDetection_ML\data\raw\WEDA-FALL\dataset\50Hz\D01\U01_R01_accel.csv")
file_path_gyro = Path(r"C:\Users\Torezhan\projects\FallDetection_ML\data\raw\WEDA-FALL\dataset\50Hz\D01\U01_R01_gyro.csv")

# Load the uploaded CSV files
df_acc = pd.read_csv(file_path_accel)
df_gyr = pd.read_csv(file_path_gyro)

# Time vectors (already in seconds)
t_acc = df_acc['accel_time_list']
t_gyr = df_gyr['gyro_time_list']

# Calculate total duration
duration = t_acc.iloc[-1] - t_acc.iloc[0]
print(f"Total Trial Duration: {duration:.2f} seconds")

# Calculate Signal Vector Magnitudes
svm_acc = np.sqrt(df_acc['accel_x_list']**2 + df_acc['accel_y_list']**2 + df_acc['accel_z_list']**2)
svm_gyr = np.sqrt(df_gyr['gyro_x_list']**2 + df_gyr['gyro_y_list']**2 + df_gyr['gyro_z_list']**2)

# Create Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Panel 1: Accelerometer
axes[0].plot(t_acc, df_acc['accel_x_list'], label='ax (X-axis)')
axes[0].plot(t_acc, df_acc['accel_y_list'], label='ay (Y-axis)')
axes[0].plot(t_acc, df_acc['accel_z_list'], label='az (Z-axis)')
axes[0].set_title(f'Raw Accelerometer Data (Total Duration: {duration:.2f}s)')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Panel 2: Gyroscope
axes[1].plot(t_gyr, df_gyr['gyro_x_list'], label='gx (Pitch)')
axes[1].plot(t_gyr, df_gyr['gyro_y_list'], label='gy (Roll)')
axes[1].plot(t_gyr, df_gyr['gyro_z_list'], label='gz (Yaw)')
axes[1].set_title('Raw Gyroscope Data')
axes[1].set_ylabel('Angular Velocity (rad/s)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Panel 3: Total Signal Vector Magnitude
axes[2].plot(t_acc, svm_acc, label='Accel Magnitude (SVM)', color='purple')
axes[2].plot(t_gyr, svm_gyr, label='Gyro Magnitude', color='orange', alpha=0.8)
axes[2].set_title('Overall Movement Magnitude')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Magnitude')
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()