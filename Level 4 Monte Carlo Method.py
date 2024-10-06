# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:04:23 2024

@author: Nanditha sajeev
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta  # Added timedelta import
import matplotlib.pyplot as plt
import os
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Set directories for data
cat_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\catalogs\\'
cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')

# Load the catalog
try:
    catalog = pd.read_csv(cat_file)
except FileNotFoundError:
    print(f"Error: The catalog file {cat_file} was not found.")
    raise

# Select the first seismic event in the catalog
event = catalog.iloc[0]  # Change the index for different events
event_time = datetime.strptime(event['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
filename = event.filename  # Get the name of the file

# Load the corresponding MiniSEED data
data_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\data\S12_GradeA\\'
mseed_file_path = os.path.join(data_directory, f'{filename}.mseed')

try:
    stream = read(mseed_file_path)
except FileNotFoundError:
    print(f"Error: The MiniSEED file {mseed_file_path} was not found.")
    raise

# Copy the first trace and extract times and data
trace = stream.traces[0].copy()
trace_times = trace.times()
trace_data = trace.data

# Sampling frequency of our trace
df = trace.stats.sampling_rate

# Define STA/LTA parameters
sta_len = 120  # Short-term average window length in seconds
lta_len = 600  # Long-term average window length in seconds

# Run Obspy's STA/LTA to obtain a characteristic function
cft = classic_sta_lta(trace_data, int(sta_len * df), int(lta_len * df))

# Define thresholds for detecting seismic events
thr_on = 4   # Trigger 'on' threshold
thr_off = 1.5  # Trigger 'off' threshold

# Get trigger onsets and offsets
on_off = trigger_onset(cft, thr_on, thr_off)

# Determine noise levels based on the characteristic function
mean_cft = np.mean(cft)
std_cft = np.std(cft)

# Compile detection times and filenames
# File name and start time of trace
fname = filename  # Use the event filename for consistency
starttime = trace.stats.starttime.datetime  # Use the trace start time

# Initialize lists to store detection times and file names
detection_times = []
fnames = []

# Iterate through detection times and compile them
for i in range(len(on_off)):
    triggers = on_off[i]
    
    # Check if triggers exist by checking the length
    if len(triggers) > 0:  
        on_time = starttime + timedelta(seconds=trace_times[triggers[0]])
        on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
        
        # Append detection time and filename
        detection_times.append(on_time_str)
        fnames.append(fname)

# Compile dataframe of detections
detect_df = pd.DataFrame(data={
    'filename': fnames,
    'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
    'time_rel(sec)': [trace_times[triggers[0]] for triggers in on_off if len(triggers) > 0]  # Ensure we only append valid triggers
})

# Display the first few rows of the dataframe
print(detect_df.head())

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the characteristic function
ax.plot(trace_times, cft, color='blue', label='STA/LTA')

# Mark the trigger points
for trigger in on_off:
    ax.axvline(x=trace_times[trigger[0]], color='red', linestyle='--', label='Trigger On' if trigger[0] == on_off[0][0] else "")
    ax.axvline(x=trace_times[trigger[1]], color='purple', linestyle='--', label='Trigger Off' if trigger[1] == on_off[0][1] else "")

# Highlight mean and threshold levels
ax.axhline(y=mean_cft + thr_on * std_cft, color='green', linestyle=':', label='Detection Threshold')
ax.axhline(y=mean_cft + thr_off * std_cft, color='orange', linestyle=':', label='Noise Threshold')

# Customizing the plot
ax.set_xlim([min(trace_times), max(trace_times)])
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('STA/LTA', fontsize=12)
ax.set_title(f'Seismic Event Detection - {filename}', fontsize=16, fontweight='bold')
ax.legend(loc='upper left')
ax.grid()

plt.tight_layout()
plt.show()
