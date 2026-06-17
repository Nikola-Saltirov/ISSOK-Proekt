import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt

# Works whether numpy has trapezoid (>=2.0) or only the deprecated trapz (<2.0)
# (avoid getattr's default-arg form here, since that evaluates np.trapz eagerly
# even when it no longer exists)
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz


def butter_highpass_filter(data, cutoff=1.0, fs=128, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def extract_qeeg_metrics(df, fs, channel):
    """Compute absolute/relative band power and ratio metrics for one EEG channel."""
    raw_signal = df[channel].values  # FIX: select by the requested channel, not column 0

    filtered_signal = butter_highpass_filter(raw_signal, cutoff=1.0, fs=fs)

    freqs, psd = welch(filtered_signal, fs=fs, nperseg=fs * 4)

    # FIX: half-open intervals so a bin at an exact boundary (4, 8, 13 Hz)
    # is only ever counted in one band
    bands = {
        'Delta': (1.0, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
    }

    metrics = {}
    total_idx = (freqs >= 1.0) & (freqs < 30)
    total_power = _trapz(psd[total_idx], freqs[total_idx])

    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs < high)
        abs_pwr = _trapz(psd[idx], freqs[idx])
        metrics[f'Abs_{band}'] = abs_pwr
        metrics[f'Rel_{band}'] = (abs_pwr / total_power) * 100 if total_power > 0 else np.nan

    metrics['Alpha_Beta_Ratio'] = (
        metrics['Rel_Alpha'] / metrics['Rel_Beta'] if metrics['Rel_Beta'] else np.nan
    )
    metrics['Theta_Alpha_Ratio'] = (
        metrics['Rel_Theta'] / metrics['Rel_Alpha'] if metrics['Rel_Alpha'] else np.nan
    )
    metrics['Theta_Beta_Ratio'] = (
        metrics['Rel_Theta'] / metrics['Rel_Beta'] if metrics['Rel_Beta'] else np.nan
    )

    return metrics


def compare_qeeg_states(df1, df2, fs=128, channel="FP2-F4"):
    df1 = df1.copy()
    df2 = df2.copy()

    df1[channel] = pd.to_numeric(df1[channel].astype(str).str.replace(',', '.'), errors='coerce')
    df2[channel] = pd.to_numeric(df2[channel].astype(str).str.replace(',', '.'), errors='coerce')

    # FIX: only drop rows where the channel of interest is NaN, not the whole row
    # based on unrelated columns
    df1 = df1.dropna(subset=[channel])
    df2 = df2.dropna(subset=[channel])

    # FIX: use the fs that was actually passed in, not a hardcoded 128
    meditation_results = extract_qeeg_metrics(df1, fs=fs, channel=channel)
    baseline_results = extract_qeeg_metrics(df2, fs=fs, channel=channel)

    comparison_df = pd.DataFrame(
        [baseline_results, meditation_results],
        index=["Baseline (Thinking)", "Meditation"]
    ).T

    def safe_ratio(a, b):
        return a / b if b else np.nan

    ratios = {
        "Alpha_Beta_Ratio": safe_ratio(
            meditation_results["Alpha_Beta_Ratio"], baseline_results["Alpha_Beta_Ratio"]
        ),
        "Theta_Alpha_Ratio": safe_ratio(
            meditation_results["Theta_Alpha_Ratio"], baseline_results["Theta_Alpha_Ratio"]
        ),
        "Theta_Beta_Ratio": safe_ratio(
            meditation_results["Theta_Beta_Ratio"], baseline_results["Theta_Beta_Ratio"]
        ),
    }

    return {
        "comparison_table": comparison_df.to_dict(orient="index"),
        "ratios": ratios,
        "baseline": baseline_results,
        "meditation": meditation_results,
    }