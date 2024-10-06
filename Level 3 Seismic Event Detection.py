# Code by Faizan Khan

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def monte_carlo_threshold_simulation(num_simulations, mean_signal, std_signal, mean_noise, std_noise):
    thresholds = []
    
    for _ in range(num_simulations):
        observed_magnitude = np.random.normal(mean_signal, std_signal)
        noise = np.random.normal(mean_noise, std_noise)
        operational_threshold = observed_magnitude - noise
        true_threshold = operational_threshold
        thresholds.append((observed_magnitude, operational_threshold, true_threshold))
    
    return np.array(thresholds)
#Specify the directory for data here
cat_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\catalogs\\'
cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')

try:
    catalog = pd.read_csv(cat_file)
except FileNotFoundError:
    print(f"Error: The catalog file {cat_file} was not found.")
    raise

event = catalog.iloc[0]
event_time = datetime.strptime(event['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
filename = event.filename
#Specify the directory for data here
data_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\data\S12_GradeA\\'
mseed_file_path = os.path.join(data_directory, f'{filename}.mseed')

try:
    stream = read(mseed_file_path)
except FileNotFoundError:
    print(f"Error: The MiniSEED file {mseed_file_path} was not found.")
    raise

trace = stream.traces[0].copy()
trace_times = trace.times()
trace_data = trace.data

start_time = trace.stats.starttime.datetime

df = trace.stats.sampling_rate
sta_len = 120
lta_len = 600

cft = classic_sta_lta(trace_data, int(sta_len * df), int(lta_len * df))

thr_on = 2
thr_off = 1
on_off = np.array(trigger_onset(cft, thr_on, thr_off))

is_seismic = np.zeros(len(trace_data), dtype=bool)

for trigger in on_off:
    start_index = trigger[0]
    end_index = trigger[1]
    is_seismic[start_index:end_index] = True

mean_signal = 3.0
std_signal = 0.5
mean_noise = 1.0
std_noise = 0.2
num_simulations = 1000

thresholds = monte_carlo_threshold_simulation(num_simulations, mean_signal, std_signal, mean_noise, std_noise)

observed_magnitudes = thresholds[:, 0]
operational_thresholds = thresholds[:, 1]
true_thresholds = thresholds[:, 2]

bias = observed_magnitudes - true_thresholds

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(trace_times, trace_data, color='navy', label='Seismogram', linewidth=1)
ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
ax1.set_title(f'Seismic Event Detection - {filename}', fontsize=16, fontweight='bold')
ax1.grid(True, linestyle=':', linewidth=0.7)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.plot(trace_times, cft, color='orange', label='STA/LTA', linewidth=1)
ax2.axhline(y=thr_on, color='red', linestyle='--', label='Trigger On Threshold')
ax2.axhline(y=thr_off, color='purple', linestyle='--', label='Trigger Off Threshold')

ax2.set_ylabel('STA/LTA', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')

ax1.fill_between(trace_times, trace_data, where=is_seismic, color='orange', alpha=0.5, label='Seismic Activity')

plt.tight_layout()
plt.show()

if len(on_off) == 0:
    print("No seismic activity detected.")
else:
    print(f"Detected {len(on_off)} seismic events.")

print(f"Mean Bias: {np.mean(bias)}")
print(f"Standard Deviation of Bias: {np.std(bias)}")
