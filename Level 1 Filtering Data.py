# coded by Faizan Khan
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
import matplotlib.cm as cm
# Set directories for data
cat_directory = r'C:\Users\Nanditha sajeev\.spyder-py3\data\lunar\training\catalogs\\'
cat_file = os.path.join(cat_directory, 'apollo12_catalog_GradeA_final.csv')

try:
    catalog = pd.read_csv(cat_file)
except FileNotFoundError:
    print(f"Error: The catalog file {cat_file} was not found.")
    raise

event = catalog.iloc[0]
event_time = datetime.strptime(event['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
relative_time = event['time_rel(sec)']
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
arrival_relative = (event_time - start_time).total_seconds()

minfreq = 0.5
maxfreq = 1.0

stream_filtered = stream.copy()
stream_filtered.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
trace_filtered = stream_filtered.traces[0].copy()
trace_times_filtered = trace_filtered.times()
trace_data_filtered = trace_filtered.data

f, t, sxx = signal.spectrogram(trace_data_filtered, fs=trace.stats.sampling_rate, nperseg=1024)

fig = plt.figure(figsize=(10, 12))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(trace_times_filtered, trace_data_filtered, color='royalblue', linewidth=1.5)

ax1.axvline(x=arrival_relative, color='darkred', linewidth=2.5, linestyle='--', label='Detection')

ax1.grid(True, which='both', linestyle=':', linewidth=0.7)

ax1.set_xlim([min(trace_times_filtered), max(trace_times_filtered)])
ax1.set_ylabel('Filtered Velocity (m/s)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax1.set_title(f'Seismic Event: {filename} (Filtered)', fontweight='bold', fontsize=16, color='navy')

ax1.legend(loc='upper left', fontsize=10)

ax2 = plt.subplot(2, 1, 2)
spectrogram_plot = ax2.pcolormesh(t, f, sxx, cmap=cm.plasma, shading='gouraud', vmax=5e-17)

ax2.set_xlim([min(trace_times_filtered), max(trace_times_filtered)])
ax2.set_xlabel(f'Time (s)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Frequency (Hz)', fontweight='bold', fontsize=12)
ax2.axvline(x=arrival_relative, color='darkred', linewidth=2.5, linestyle='--')

cbar = plt.colorbar(spectrogram_plot, ax=ax2, orientation='horizontal', pad=0.2, fraction=0.05)
cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold', fontsize=10, color='darkblue')

plt.tight_layout()
plt.show()
