from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt 
import pywt
from scipy import signal
# Signals is of shape (time, leads, instance)

def highpass_filter(signals, cutoff=0.5, fs=1000, order=2): # order for highpass usually 2-4
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    filtered_signals = sosfiltfilt(sos, signals, axis=1)
    return filtered_signals

def notch_filter(signals, freq=50.0, fs=1000, Q=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, Q)
    filtered_signals = filtfilt(b, a, signals, axis=1) # This can be done with sosfiltfilt but for simplicity.
    return filtered_signals

def lowpass_filter(signals, cutoff=150.0, fs=1000, order=2): # order for lowpass usually 4-6
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    filtered_signals = sosfiltfilt(sos, signals, axis=1)
    return filtered_signals

def wavelet_denoise(ecg_data, wavelet='db6', level=4, epsilon=1e-10):
    denoised_ecg = np.zeros_like(ecg_data)
    for i in range(ecg_data.shape[1]):
        coeffs = pywt.wavedec(ecg_data[:, i], wavelet, level=level)
        median_abs = np.median(np.abs(coeffs[-level]))
        if median_abs == 0:
            threshold = 0
        else:
            threshold = median_abs / 0.6745
        
        def safe_threshold(c):
            thresholded = pywt.threshold(c, threshold, mode='soft')
            return np.where(np.isfinite(thresholded) & (np.abs(c) > epsilon), thresholded, 0)
        
        new_coeffs = [coeffs[0]] + [safe_threshold(c) for c in coeffs[1:]]
        denoised_ecg[:, i] = pywt.waverec(new_coeffs, wavelet)
    
    # Replace any remaining NaN or inf values with zeros
    denoised_ecg = np.nan_to_num(denoised_ecg, nan=0.0, posinf=0.0, neginf=0.0)
    return denoised_ecg

def advanced_ecg_filter(ecg_data, fs=500, notch_freqs=[50, 60], highcut=100.0):
    filtered_ecg = ecg_data.copy()
    
    quality_factor = 30.0
    for notch_freq in notch_freqs:
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
        filtered_ecg = signal.filtfilt(b_notch, a_notch, filtered_ecg, axis=0)

    lowcut = 0.5
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 4

    b_band, a_band = signal.butter(order, [low, high], btype='band')
    filtered_ecg = signal.filtfilt(b_band, a_band, filtered_ecg, axis=0)

    baseline_cutoff = 0.05
    baseline_low = baseline_cutoff / nyquist
    b_baseline, a_baseline = signal.butter(order, baseline_low, btype='high')
    filtered_ecg = signal.filtfilt(b_baseline, a_baseline, filtered_ecg, axis=0)

    return filtered_ecg
