from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt 

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
