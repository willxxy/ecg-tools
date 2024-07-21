import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


ecg_data = np.load('./test_Ecg.npy') # (12, 1000)

min_val = np.min(ecg_data)
max_val = np.max(ecg_data)
normalized_ecg = (ecg_data - min_val) / (max_val - min_val)

lead_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ecg_reordered = normalized_ecg[lead_order]

time_scale = 25  # mm/s
voltage_scale = 10  # mm/mV
sampling_rate = 500  # Hz
time = np.arange(ecg_reordered.shape[1]) / sampling_rate  # Time in seconds

fig, axs = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
axs = axs.flatten()
lead_names = ['I', 'aVR', 'V1', 'V4',
              'II', 'aVL', 'V2', 'V5',
              'III', 'aVF', 'V3', 'V6']

for i, (ax, lead_name) in enumerate(zip(axs, lead_names)):
    ax.plot(time, ecg_reordered[i], 'k-', linewidth=0.5)
    
    ax.set_title(lead_name, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, which='both', color='r', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks(np.arange(0, 2.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))  # Adjusted for normalized data
    ax.set_xticks(np.arange(0, 2.02, 0.04), minor=True)
    ax.set_yticks(np.arange(0, 1.02, 0.04), minor=True)  # Adjusted for normalized data

plt.tight_layout()
plt.savefig("./test_Ecg.png")
