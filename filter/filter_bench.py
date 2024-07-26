import numpy as np
from ptb_utils import filter_all, highpass_filter, notch_filter, lowpass_filter
import time
import random
import psutil
import os
import matplotlib.pyplot as plt

def set_cpu_affinity():
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0])  # Use only the first CPU core

def run_benchmark(num_instances, single_instance, num_trials=10, num_warmup=5):
    total_times = {'highpass': 0, 'notch': 0, 'lowpass': 0}
    signal = np.tile(single_instance, (num_instances, 1, 1))
    
    for _ in range(num_warmup):
        _ = filter_all(signal)
    
    for _ in range(num_trials):
        start_time = time.perf_counter()
        highpass_signal = highpass_filter(signal)
        total_times['highpass'] += time.perf_counter() - start_time

        start_time = time.perf_counter()
        notch_signal = notch_filter(highpass_signal)
        total_times['notch'] += time.perf_counter() - start_time

        start_time = time.perf_counter()
        lowpass_signal = lowpass_filter(notch_signal)
        total_times['lowpass'] += time.perf_counter() - start_time
    
    avg_times = {k: v / num_trials for k, v in total_times.items()}
    total_avg_time = sum(avg_times.values())
    time_per_instance = total_avg_time / num_instances
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    return avg_times, total_avg_time, time_per_instance, memory_usage

set_cpu_affinity()

instance_sizes = [100, 200, 400, 800, 1600, 3200]
random.shuffle(instance_sizes)

single_instance = np.random.random((1, 1000, 12))
results = []

for num_instances in instance_sizes:
    print(f"Processing {num_instances} instances")
    avg_times, total_avg_time, time_per_instance, memory_usage = run_benchmark(num_instances, single_instance)
    results.append((num_instances, avg_times, total_avg_time, time_per_instance, memory_usage))
    print(f"Average times for {num_instances} instances:")
    for filter_name, avg_time in avg_times.items():
        print(f"  {filter_name}: {avg_time:.6f} seconds")
    print(f"Total average time: {total_avg_time:.6f} seconds")
    print(f"Time per instance: {time_per_instance:.6f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    print()

results.sort(key=lambda x: x[0])

instance_sizes, avg_times_list, total_avg_times, times_per_instance, memory_usages = zip(*results)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

ax1.plot(instance_sizes, total_avg_times, marker='o')
ax1.set_xlabel('Number of Instances')
ax1.set_ylabel('Total Time (seconds)')
ax1.set_title('Total Filter Time vs Number of Instances')
ax1.grid(True)

ax2.plot(instance_sizes, times_per_instance, marker='o')
ax2.set_xlabel('Number of Instances')
ax2.set_ylabel('Time per Instance (seconds)')
ax2.set_title('Time per Instance vs Number of Instances')
ax2.grid(True)

ax3.plot(instance_sizes, memory_usages, marker='o')
ax3.set_xlabel('Number of Instances')
ax3.set_ylabel('Memory Usage (MB)')
ax3.set_title('Memory Usage vs Number of Instances')
ax3.grid(True)

for filter_name in ['highpass', 'notch', 'lowpass']:
    filter_times = [times[filter_name] for times in avg_times_list]
    ax4.plot(instance_sizes, filter_times, marker='o', label=filter_name)
ax4.set_xlabel('Number of Instances')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Individual Filter Times vs Number of Instances')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('filter_all_performance.png')
