#code by Faizan Khan

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

cat_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\catalogs\\'
cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')

try:
    catalog = pd.read_csv(cat_file)
except FileNotFoundError:
    print(f"Error: The catalog file {cat_file} was not found.")
    raise

event = catalog.iloc[0]
event_time = datetime.strptime(event['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
filename = event.filename

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

thr_on = 4
thr_off = 1.5

on_off = trigger_onset(cft, thr_on, thr_off)

detection_times = []
fnames = []

for triggers in on_off:
    if len(triggers) > 0:
        on_time = start_time + timedelta(seconds=trace.times()[triggers[0]])
        on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
        detection_times.append(on_time_str)
        fnames.append(filename)

detect_df = pd.DataFrame(data={
    'filename': fnames,
    'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
    'time_rel(sec)': [trace.times()[triggers[0]] for triggers in on_off if len(triggers) > 0]
})

print(detect_df.head())

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(trace_times, cft, color='navy', linewidth=1.5)
for trigger in on_off:
    ax1.axvline(x=trace_times[trigger[0]], color='green', linestyle=':', label='Trigger On' if trigger[0] == on_off[0][0] else "")
    ax1.axvline(x=trace_times[trigger[1]], color='orange', linestyle=':', label='Trigger Off' if trigger[1] == on_off[0][1] else "")

ax1.set_xlim([min(trace_times), max(trace_times)])
ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('STA/LTA Characteristic', fontsize=12, fontweight='bold')
ax1.set_title(f'STA/LTA Detection - {filename}', fontweight='bold', fontsize=16)

ax1.grid(True, linestyle=':', linewidth=0.7)
ax1.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
