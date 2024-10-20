import numpy as np
import pywt
from scipy import signal

def extract_features(ecg, sampling_rate=250):
    features = []
    
    for lead in range(ecg.shape[0]):
        lead_signal = ecg[lead, :]
        
        # Basic statistical features
        features.extend([
            np.mean(lead_signal),
            np.std(lead_signal),
            np.max(lead_signal),
            np.min(lead_signal),
            np.median(lead_signal),
            np.percentile(lead_signal, 25),
            np.percentile(lead_signal, 75)
        ])
        
        # Frequency domain features
        freqs, psd = signal.welch(lead_signal, fs=sampling_rate, nperseg=1024)
        total_power = np.sum(psd)
        features.extend([
            total_power,  # Total power
            np.max(psd),  # Peak frequency power
            freqs[np.argmax(psd)],  # Dominant frequency
        ])
        
        # Spectral centroid with NaN handling
        if total_power > 0:
            spectral_centroid = np.sum(freqs * psd) / total_power
        else:
            spectral_centroid = 0  # or another appropriate default value
        features.append(spectral_centroid)
        
        peaks, _ = signal.find_peaks(lead_signal, height=0.5*np.max(lead_signal), distance=0.2*sampling_rate)
        if len(peaks) > 1:
            # Heart rate
            rr_intervals = np.diff(peaks) / sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)
            features.append(heart_rate)
            
            # Heart rate variability
            hrv = np.std(rr_intervals)
            features.append(hrv)
            
            # QRS duration (simplified)
            qrs_duration = np.mean([find_qrs_duration(lead_signal, peak, sampling_rate) for peak in peaks])
            features.append(qrs_duration)
        else:
            features.extend([0, 0, 0])  # Placeholder values if no peaks found
        
        # T-wave features (simplified)
        t_wave_amp = find_t_wave_amplitude(lead_signal, peaks)
        features.append(t_wave_amp)
        
        # ST segment features (simplified)
        st_deviation = find_st_deviation(lead_signal, peaks, sampling_rate)
        features.append(st_deviation)
        
        coeffs = pywt.wavedec(lead_signal, 'db4', level=5)
        features.extend([np.mean(np.abs(c)) for c in coeffs])
        
        # Non-linear features
        features.append(np.mean(np.abs(np.diff(lead_signal))))  # Average absolute difference
        features.append(np.sqrt(np.mean(np.square(np.diff(lead_signal)))))  # Root mean square of successive differences
    
    return np.array(features)


def find_qrs_duration(ecg, peak, sampling_rate):
    # Simplified QRS duration estimation
    window = int(0.1 * sampling_rate)  # 100 ms window
    start = max(0, peak - window)
    end = min(len(ecg), peak + window)
    qrs_segment = ecg[start:end]
    return np.sum(np.abs(qrs_segment) > 0.1 * np.max(qrs_segment)) / sampling_rate

def find_t_wave_amplitude(ecg, peaks):
    if len(peaks) < 2:
        return 0
    t_wave_region = ecg[peaks[-2]:peaks[-1]]
    return np.max(t_wave_region) - np.min(t_wave_region)

def find_st_deviation(ecg, peaks, sampling_rate):
    if len(peaks) < 2:
        return 0
    st_point = peaks[-1] + int(0.08 * sampling_rate)  # 80 ms after R peak
    if st_point < len(ecg):
        return ecg[st_point] - ecg[peaks[-1]]
    return 0
