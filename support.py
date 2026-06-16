import pandas as pd
from scipy.signal import welch, butter, filtfilt
import numpy as np

def butter_highpass_filter(data, cutoff=1.0, fs=128, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def extract_qeeg_metrics(signal, fs):
    raw_signal = np.asarray(signal).flatten()

    filtered_signal = butter_highpass_filter(raw_signal, cutoff=1.0, fs=fs)


    freqs, psd = welch(filtered_signal, fs=fs, nperseg=fs*4)

    bands = {
        'Delta': (1.0, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }

    metrics = {}
    total_idx = (freqs >= 1.0) & (freqs <= 30)
    total_power = np.trapezoid(psd[total_idx], freqs[total_idx])

    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        abs_pwr = np.trapezoid(psd[idx], freqs[idx])
        metrics[f'Abs_{band}'] = abs_pwr
        metrics[f'Rel_{band}'] = (abs_pwr / total_power) * 100

    metrics['Alpha_Beta_Ratio'] = metrics['Rel_Alpha'] / metrics['Rel_Beta']
    metrics['Theta_Alpha_Ratio'] = metrics['Rel_Theta'] / metrics['Rel_Alpha']
    metrics['Theta_Beta_Ratio'] = metrics['Rel_Theta'] / metrics['Rel_Beta']

    return metrics

def calculate_region_average(df, channels, metric):
    existing_channels = [ch for ch in channels if ch in df.index]

    if len(existing_channels) == 0:
        raise ValueError("None of the specified channels exist in the DataFrame.")

    return df.loc[existing_channels, metric].mean()

def calculate_alpha_regions(df):
    frontal = ['Fp1','Fp2','F3','F4','F7','F8','Fz']
    central = ['C3','C4','Cz']
    parietal = ['P3','P4','Pz']
    occipital = ['O1','O2']
    temporal = ['T3','T4','T5','T6']

    return [
        {
            "Region": "Frontal",
            "Rel_Alpha": df.loc[frontal, "Rel_Alpha"].mean(),
            "Rel_Beta": df.loc[frontal, "Rel_Beta"].mean(),
            "Rel_Theta": df.loc[frontal, "Rel_Theta"].mean(),
        },
        {
            "Region": "Central",
            "Rel_Alpha": df.loc[central, "Rel_Alpha"].mean(),
            "Rel_Beta": df.loc[central, "Rel_Beta"].mean(),
            "Rel_Theta": df.loc[central, "Rel_Theta"].mean(),
        },
        {
            "Region": "Parietal",
            "Rel_Alpha": df.loc[parietal, "Rel_Alpha"].mean(),
            "Rel_Beta": df.loc[parietal, "Rel_Beta"].mean(),
            "Rel_Theta": df.loc[parietal, "Rel_Theta"].mean(),
        },
        {
            "Region": "Occipital",
            "Rel_Alpha": df.loc[occipital, "Rel_Alpha"].mean(),
            "Rel_Beta": df.loc[occipital, "Rel_Beta"].mean(),
            "Rel_Theta": df.loc[occipital, "Rel_Theta"].mean(),
        },
        {
            "Region": "Temporal",
            "Rel_Alpha": df.loc[temporal, "Rel_Alpha"].mean(),
            "Rel_Beta": df.loc[temporal, "Rel_Beta"].mean(),
            "Rel_Theta": df.loc[temporal, "Rel_Theta"].mean(),
        }
    ]

def extract_band_ratios_from_csv(csv_file, fs=500, channel_names=None):
    if channel_names is not None:
        df = pd.read_csv(csv_file, header=None, names=channel_names)
    else:
        df = pd.read_csv(csv_file)

    df.columns = df.columns.str.strip()

    results = []

    for channel in df.columns:
        try:
            signal = pd.to_numeric(
                df[channel].astype(str).str.replace(',', '.'),
                errors='coerce'
            ).dropna().values

            if len(signal) == 0:
                continue

            metrics = extract_qeeg_metrics(signal, fs)

            results.append({
                'Channel': channel,
                'Rel_Alpha': metrics['Rel_Alpha'],
                'Rel_Theta': metrics['Rel_Theta'],
                'Rel_Beta': metrics['Rel_Beta'],
                'Alpha_Beta_Ratio': metrics['Alpha_Beta_Ratio'],
                'Theta_Alpha_Ratio': metrics['Theta_Alpha_Ratio'],
                'Theta_Beta_Ratio': metrics['Theta_Beta_Ratio']
            })

        except Exception as e:
            print(f"Error processing {channel}: {e}")

    return pd.DataFrame(results)

