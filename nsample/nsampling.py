import numpy as np
from scipy import interpolate
def nsample_ecg(ecg_data, orig_fs, target_fs):
    num_samples, num_leads = ecg_data.shape
    duration = num_samples / orig_fs
    t_original = np.linspace(0, duration, num_samples, endpoint=True)
    t_target = np.linspace(0, duration, int(num_samples * target_fs / orig_fs), endpoint=True)
    
    downsampled_data = np.zeros((len(t_target), num_leads))
    for lead in range(num_leads):
        f = interpolate.interp1d(t_original, ecg_data[:, lead], kind='cubic', bounds_error=False, fill_value="extrapolate")
        downsampled_data[:, lead] = f(t_target)
    return downsampled_data
