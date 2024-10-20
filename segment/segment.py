import numpy as np

def segment_ecg(ecg_data, text_data, seg_len):
    time_length, _ = ecg_data.shape
    num_segments = time_length // seg_len
    
    ecg_data_segmented = []
    text_data_segmented = []
    
    for i in range(num_segments):
        start_idx = i * seg_len
        end_idx = (i + 1) * seg_len
        ecg_data_segmented.append(ecg_data[start_idx:end_idx, :])
        text_data_segmented.append(text_data)
    
    return np.array(ecg_data_segmented), text_data_segmented
